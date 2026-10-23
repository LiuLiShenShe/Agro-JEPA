# S0 — DINO-WM API Map (real call chain and interface contracts)

Companion to `reports/s0_upstream_audit.md`.
Line numbers refer to upstream HEAD `0a9492fa12044b852ae9e001cc74604b79c8bb0c`
(the only local change is the `:main` hub ref in `models/dino.py`, commit `c7417ce`).
Shapes marked ✅ were produced by `tools/s0_forward_smoke.py` on this host.

## 0. Reading guide

| Need | Read this |
|---|---|
| where data enters | §2 `Trainer.__init__` dataset block, `TrajSlicerDataset` |
| where features are computed | §3 `VWorldModel.encode_obs` → `DinoV2Encoder` |
| where the future is predicted | §3 `VWorldModel.predict` → `ViTPredictor` |
| where the loss is defined | §4 `VWorldModel.forward` |
| where actions condition the model | §3 causal-mask + action-token explanation, §6.C |
| what to change for agro | §6 (interface list), §7 (module mapping) |
| what is unsupported | §8 |

## 1. Training call chain (verified by execution)

```
python train.py --config-name train.yaml
└─ @hydra.main(config_path="conf", config_name="train")              train.py:815
   └─ Trainer.__init__                                              train.py:30
      ├─ cfg.saved_folder = os.getcwd()                             train.py:33
      ├─ Accelerator(log_with="wandb")                              train.py:60
      ├─ wandb.init(project="dino_wm", config=...)                  train.py:100
      ├─ hydra.utils.call(cfg.env.dataset,                          train.py:115
      │        num_hist, num_pred, frameskip)
      │   └─ datasets/<task>_dset.py::load_*_slice_train_val
      │      ├─ <Task>Dataset(...)  → torch.load states/actions/seq_lengths
      │      ├─ get_train_val_sliced(...)                           traj_dset.py:138
      │      │   ├─ split_traj_datasets → random_split_traj         traj_dset.py:126/103
      │      │   │   └─ TrajSubset(dataset, indices)
      │      │   └─ TrajSlicerDataset(train/val, num_frames=num_hist+num_pred, frameskip)
      │      └─ returns (datasets{tr,va}, traj_dsets{tr,va})
      ├─ DataLoader(batch_size=gpu_batch_size, shuffle=False)       train.py:125
      ├─ accelerator.prepare(train_loader, val_loader)              train.py:138
      ├─ init_models()                                              train.py:205
      │   ├─ encoder = instantiate(cfg.encoder)                     train.py:213
      │   │   └─ DinoV2Encoder.__init__                             models/dino.py:7
      │   │       └─ torch.hub.load("facebookresearch/dinov2:main", name)
      │   ├─ freeze encoder if not train_encoder                    train.py:216
      │   ├─ proprio_encoder = instantiate(cfg.proprio_encoder,     train.py:220
      │   │        in_chans=dataset.proprio_dim, emb_dim=proprio_emb_dim)
      │   ├─ action_encoder = instantiate(cfg.action_encoder,       train.py:229
      │   │        in_chans=dataset.action_dim, emb_dim=action_emb_dim)
      │   ├─ num_patches = 1 if encoder.latent_ndim==1 else         train.py:244
      │   │        (img_size // 16)**2; +2 if concat_dim==0
      │   ├─ predictor = instantiate(cfg.predictor,                 train.py:254
      │   │        num_patches, num_frames=num_hist,
      │   │        dim=encoder.emb_dim + (proprio+action)*concat_dim)
      │   ├─ decoder = instantiate(cfg.decoder, emb_dim=encoder.emb_dim)  train.py:285
      │   ├─ accelerator.prepare(encoder, predictor, decoder)       train.py:292
      │   └─ model = instantiate(cfg.model, ...)                    train.py:295
      │        └─ VWorldModel.__init__                              visual_world_model.py:7
      ├─ init_optimizers()                                          train.py:309
      │   ├─ Adam(encoder, lr=encoder_lr)
      │   ├─ AdamW(predictor, lr=predictor_lr)                      (if has_predictor)
      │   ├─ AdamW(action_encoder.parameters() + proprio_encoder.parameters(),
      │   │        lr=action_encoder_lr)                            (if has_predictor)
      │   └─ Adam(decoder, lr=decoder_lr)                           (if has_decoder)
      └─ run()                                                      train.py:360
         ├─ train()                                                 train.py:442
         │   for obs, act, state in loader:
         │     z_pred, visual_pred, visual_rec, loss, comps = model(obs, act)
         │     accelerator.backward(loss)
         │     encoder/predictor/decoder/action(+proprio) optimizer.step()
         ├─ val()                                                   train.py:537
         │   ├─ openloop_rollout(train_traj_dset) and (val_traj_dset)  train.py:632
         │   └─ forward-only loss over val loader
         ├─ logs_flash()  → wandb
         └─ save_ckpt()  → checkpoints/model_{epoch}.pth, model_latest.pth
```

## 2. Data contract

### 2.1 What a dataset class must provide

`TrajDataset` is an `abc.ABC` with one abstract method (`traj_dset.py:25-31`), and the
slicer reads three attributes:

| Member | Shape / type | Consumed at |
|---|---|---|
| `get_seq_length(idx) -> int` | scalar | `traj_dset.py:66` |
| `__getitem__(idx)` | `(obs, act, state, env_info)` | `traj_dset.py:94` |
| `obs` | `dict[str, Tensor]`, keys `"visual"`, `"proprio"` | slicer slices every key |
| `obs["visual"]` | `(T, 3, H, W)` float, already `/255` + transform | `traj_dset.py:96` |
| `obs["proprio"]` | `(T, P)` float, normalized in dataset | `traj_dset.py:96` |
| `act` | `(T, A)` float, normalized in dataset | `traj_dset.py:98` |
| `state` | `(T, S)` float | sliced but unused in the loss |
| `action_dim`, `proprio_dim`, `state_dim` | int attributes | `train.py:222/231` |
| `action_mean/std`, `proprio_mean/std`, `state_mean/std` | tensors | `plan.py` preprocessor |

`env_info` (4th return value) is task-specific and ignored by training; `plan.py` uses it to
configure the simulator.

### 2.2 What the slicer emits (the actual training batch)

`TrajSlicerDataset.__init__` (`traj_dset.py:53-83`):
- `num_frames = num_hist + num_pred` (the loader is called with `num_hist + num_pred`,
  e.g. `point_maze_dset.py:131`);
- one slice per `(traj, start)` with `start ∈ [0, T - num_frames*frameskip]`;
- `action_dim = dataset.action_dim * frameskip` (line 78-81);
- slices are globally permuted with numpy at construction; `DataLoader(shuffle=False)`.

`__getitem__` (`traj_dset.py:92-100`):
```
obs[k] = v[start:end:frameskip]          # (num_frames, ...)
state  = state[start:end:frameskip]
act    = act[start:end]                  # (num_frames*frameskip, A)
act    = rearrange(act, "(n f) d -> n (f d)", n=num_frames)   # (num_frames, A*frameskip)
```

### 2.3 Verified shape ledger

Config: `num_hist=3, num_pred=1, frameskip=5, img_size=224, P=4, A=2, batch b=2`.
`T = num_hist + num_pred = 4` observation slots spanning `4*5 = 20` raw frames.

| Tensor | Shape | Where |
|---|---|---|
| `obs["visual"]` | `(2, 4, 3, 224, 224)` | DataLoader |
| `obs["proprio"]` | `(2, 4, 4)` | DataLoader |
| `act` | `(2, 4, 10)` = `(b, T, A*frameskip)` | DataLoader |
| encoder input | `(8, 3, 196, 196)` | after `rearrange` + `Resize(196)` ✅ |
| DINO patch tokens | `(8, 196, 384)` ViT-S / `(8, 196, 768)` ViT-B ✅ | `DinoV2Encoder.forward` |
| `z_visual` | `(2, 4, 196, 384)` / `(2, 4, 196, 768)` | `encode_obs` |
| `proprio_emb` | `(2, 4, 10)` | `ProprioceptiveEmbedding(4→10)` |
| `act_emb` | `(2, 4, 10)` | `ProprioceptiveEmbedding(10→10)` |
| `z` (concat_dim=1) | `(2, 4, 196, 404)` / `(2, 4, 196, 788)` ✅ | `VWorldModel.encode` |
| `z_src` | `(2, 3, 196, 404)` | `z[:, :num_hist]` |
| `z_tgt` | `(2, 3, 196, 404)` | `z[:, num_pred:]` |
| predictor input | `(2, 588, 404)` | `(t p)` flatten |
| predictor output | `(2, 588, 404)` → `(2, 3, 196, 404)` | `predict` |
| decoder output | `(12, 3, 224, 224)` → `(2, 3, 3, 224, 224)` | VQVAE ✅ |
| `visual_rec` | `(2, 4, 3, 224, 224)` | decoder of `z` ✅ |

`num_patches = (img_size // 16)**2 = 196` for both ViT-S and ViT-B because patch size is 14
and the encoder is resized to `(224//16)*14 = 196`. ViT-B changes the **feature dim only**.

## 3. Model contract

### 3.1 `VWorldModel.encode(obs, act)` (`visual_world_model.py:91-110`)

```python
z_dct = self.encode_obs(obs)        # visual: (b,t,196,E), proprio: (b,t,10)
act_emb = self.encode_act(act)      # (b,t,10)
# concat_dim == 1: tile proprio/action along the FEATURE axis, then concat
z = cat([visual (b,t,196,E), proprio_repeated (b,t,196,10), act_repeated (b,t,196,10)], dim=3)
# -> (b,t,196,E+20)
# concat_dim == 0: add two extra TOKENS instead:
z = cat([visual (b,t,196,E), proprio.unsqueeze(2) (b,t,1,10), act.unsqueeze(2) (b,t,1,10)], dim=2)
# -> only valid if 10 == E  (see audit §6.6)
```

`encode_obs` (`:120-134`) flattens `(b,t)` into the batch, applies
`encoder_transform = Resize(196)` (`:55-61`), calls `encoder.forward`, and unflattens.

### 3.2 `VWorldModel.predict(z)` (`:136-147`)

`(b,t,p,d) → (b, t*p, d) → ViTPredictor → (b,t,p,d)`.
`ViTPredictor.forward` (`vit.py:115-120`) adds a learned positional embedding
`(1, num_hist*num_patches, dim)` and runs `depth=6`, `heads=16`, `dim_head=64`, `mlp=2048`
transformer blocks.

### 3.3 Causal mask — how actions condition the future

`generate_mask_matrix(num_patches, num_frames)` (`vit.py:13-21`) builds a block-causal mask:
a token in frame block `i` attends to all blocks `≤ i` (all tokens inside its own block are
visible). With `concat_dim=1` the action is a feature appended to every patch token of frame
`t`, so it can influence frame `t` and all later frames, but **not** earlier frames.
With `concat_dim=0` the action is the last token of frame `t`, so frame `t`'s visual tokens
cannot see it (they precede it in the block), but frame `t+1` can. This is the entire
mechanism by which a robot action (or a weather forcing vector) enters the predictor.

The mask is created at construction with module-global `NUM_FRAMES/NUM_PATCHES`
(`vit.py:7-8, 106-108`) and moved to `cuda` (`:58`); see audit §6.3.

### 3.4 `VWorldModel.rollout(obs_0, act)` (`:284-309`)

```
num_obs_init = obs_0["visual"].shape[1]
z = encode(obs_0, act[:, :num_obs_init])          # context
for t in range(len(act) - num_obs_init):
    z_pred = predict(z[:, -num_hist:])            # (b, num_hist, p, d)
    z_new  = z_pred[:, -1:]                       # next frame
    z_new  = replace_actions_from_z(z_new, act[:, num_obs_init+t])   # overwrite action channel
    z = cat([z, z_new], dim=1)
z_pred = predict(z[:, -num_hist:]); z = cat([z, z_pred[:, -1:]], dim=1)
return separate_emb(z)
```

Verified: 3 context frames + 5 actions → `z (1,9,196,788)` (3 + 5 + 1). The extra final
`predict` produces the frame after the last supplied action. Actions are **inputs**, never
predicted (the loss excludes their columns). `rollout` ignores `num_pred` and `frameskip`.

### 3.5 Where proprio and actions are used

| Signal | Encoding | Supervised? | Used at rollout? |
|---|---|---|---|
| visual (DINO) | `DinoV2Encoder` | no (frozen) | recomputed only for the context; predicted afterwards |
| proprio | `ProprioceptiveEmbedding` (Conv1d k=1) | **yes**, `z_proprio_loss` (and included in `z_loss`) | predicted from the context; needed as input for context frames |
| action | `ProprioceptiveEmbedding` (Conv1d k=1) | **no** (loss slices it off) | supplied per step via `replace_actions_from_z` |

## 4. Loss, optimizers, checkpoint

### 4.1 Loss (`VWorldModel.forward`, `:189-271`)

With `num_hist=3, num_pred=1` and `concat_dim=1`:

```
z_src = z[:, :3];  z_tgt = z[:, 1:];  z_pred = predict(z_src)
z_loss       = MSE(z_pred[..., :-10],  z_tgt[..., :-10].detach())   # visual + proprio
z_visual_loss= MSE(z_pred[..., :-20],  z_tgt[..., :-20].detach())   # logged only
z_proprio_loss=MSE(z_pred[..., -20:-10], z_tgt[..., -20:-10].detach())  # logged only
if decoder:  decode(z_pred.detach()) vs visual_tgt   -> recon + 0.25 * vq
             decode(z.detach())      vs obs["visual"] -> recon + 0.25 * vq
loss = z_loss (+ decoder losses if has_decoder)
```

`decoder_latent_loss_weight = 0.25`; `emb_criterion = MSELoss`; `decoder_criterion = MSELoss`.
The action columns are excluded from **all** latent losses. `z_visual_loss` and
`z_proprio_loss` are logged but already contained in `z_loss`.

The target is shifted by `num_pred` frames: output position `i` is trained to be the embedding
at `i + num_pred`. This is where an explicit Δt / irregular-time signal would have to enter
(output `i` predicts a variable-length interval).

### 4.2 Optimizers (`Trainer.init_optimizers`, `:309-338`)

| Module | Optimizer | LR key | Stepped when |
|---|---|---|---|
| encoder | Adam | `training.encoder_lr` (1e-6) | `model.train_encoder` |
| predictor | AdamW | `training.predictor_lr` (5e-4) | `cfg.has_predictor and model.train_predictor` |
| action + proprio | AdamW (one optimizer over both) | `training.action_encoder_lr` (5e-4) | same as predictor |
| decoder | Adam | `training.decoder_lr` (3e-4) | `cfg.has_decoder and model.train_decoder` |

Note the action and proprio encoders are only optimized when `has_predictor` is true.

### 4.3 Checkpoint

`_keys_to_save` (`:155-169`) = `epoch` + `encoder`(+opt) if `train_encoder` +
`predictor`(+opt) if `train_predictor and has_predictor` + `decoder`(+opt) if `train_decoder`
+ `action_encoder` + `proprio_encoder`. Objects are saved whole (not `state_dict`).
The encoder is not saved when frozen. `load_ckpt` (`:197-203`) overwrites `Trainer` attributes.
See audit §6.5/§6.13/§6.15 for the torch≥2.6 and `has_decoder=False` caveats.

## 5. Switch truth table

| Config | Actual behavior (verified unless marked) |
|---|---|
| `has_decoder=True` (default) | decoder instantiated, recon losses added, image metrics + plots run, rollout can decode |
| `has_decoder=False` | `decoder=None`; forward returns `None` visuals; loss = latent only; **no** `eval_images`; rollout skips decode; `accelerator.prepare(..., None)` works ✅; but `save_ckpt` crashes if `model.train_decoder=True` (audit §6.13) |
| `has_predictor=False` | no predictor/optimizer; `forward` returns `z_pred=None`; only decoder recon is trained; `val()` skips rollout |
| `train_encoder=True` | encoder unfrozen, Adam lr=1e-6, encoder saved in ckpt |
| `train_predictor=False` | predictor frozen (requires_grad=False) but still forward; optimizer built but not stepped |
| `concat_dim=1` (default) | proprio/action are per-frame feature vectors appended to every patch; `dim = E+proprio+action` |
| `concat_dim=0` | proprio/action become 2 extra tokens; requires `proprio_emb_dim == action_emb_dim == E` (audit §6.6) |
| `num_pred=1` (default) | target is the next observation slot |
| `num_pred>1` | shape-valid with `T=num_hist+num_pred`; target is `num_pred` slots ahead; rollout unaffected ✅ |
| planning eval | only if `plan_settings.plan_cfg_path != null`; imports `plan.py`/gym/mujoco lazily |

## 6. Interface modification list for Agro-JEPA

Each item names the exact upstream hook. Nothing here is implemented in S0.

### A. Data fields and shapes

Upstream expects exactly `obs["visual"]` `(b,T,3,H,W)` and `obs["proprio"]` `(b,T,P)`, plus
`act (b,T,A*frameskip)` and `state (b,T,S)`. A FIP loader must subclass
`datasets.traj_dset.TrajDataset`, expose `get_seq_length`, `__getitem__` returning
`(obs, act, state, env_info)`, and set `action_dim/proprio_dim/state_dim` plus normalization
stats. Then point `conf/env/<name>.yaml::dataset._target_` at a
`load_*_slice_train_val(transform, n_rollout, normalize_action, data_path, split_ratio, num_hist, num_pred, frameskip, ...)`
function that calls `get_train_val_sliced` (or constructs `TrajSlicerDataset` directly).
`train.py` never imports task classes by name, so no trainer change is needed for a new task.

Constraints:
- `T >= num_hist + num_pred` **slots**, i.e. `T_raw >= (num_hist+num_pred)*frameskip`
  (the short-trajectory warning is wrong; audit §6.9);
- `visual` must be square for the current `Resize(196)` path;
- `act` is required even if it carries no robot action (see C).

### B. `num_hist` / `num_pred` / `frameskip`

- The loader is called with `num_hist` and `num_pred` separately (`train.py:115-120`);
  each task's `load_*` adds them into `num_frames=num_hist+num_pred`.
- Slicing is **uniform stride** `frameskip` + a raw-frame action block of size `frameskip`
  (`traj_dset.py:96-99`). This is the main incompatibility with FIP's irregular acquisition
  dates. Three options, in increasing invasiveness:
  1. set `frameskip=1` and pre-align each window to actual dates (keeps upstream code intact);
  2. replace `TrajSlicerDataset.__getitem__` with a date-aware window builder that still emits
     the same `(obs, act, state)` tuple (all downstream code unchanged);
  3. add a per-step Δt field to `obs` and consume it in `VWorldModel.encode` (requires model
     changes; no upstream hook exists).
- `action_dim` after slicing is `base_action_dim * frameskip`; `action_emb_dim` stays a config
  constant, so changing `frameskip` does not require a config change (Conv1d accepts any
  `in_chans`), but a checkpoint trained with a different `A*frameskip` will not load.

### C. Replacing robot proprio / action

- **proprio** is the robot state in the demos and is the only *supervised* non-visual signal.
  Replace it with a normalized physiological phenotype vector (e.g. LAI, biomass, height,
  greenness) of fixed width `P`. It is predicted during rollout, so it is available only for
  the context frames; that is the intended "anchor" slot. Missing dates need a loss mask,
  which upstream does not have (see §8).
- **action** is a robot action in the demos and is **never supervised**. Replace it with an
  environmental forcing vector per interval (GDD, rainfall integral, radiation integral, ...).
  Because encoders are per-frame Conv1d(k=1), the dataset must summarize each interval into
  one vector; the alternative is to use the existing `frameskip` concatenation to stack
  sub-interval weather and let the encoder learn the projection.
- **`action_dim=0` is not supported**: `ProprioceptiveEmbedding` would build `Conv1d(0, ...)`.
  For "no action", pass a zero vector of width ≥1 (or a custom encoder returning zeros of
  `emb_dim`). If weather is the action, `A` is simply its width.
- `replace_actions_from_z` (`:273-281`) is the single function that overwrites the action
  channel during rollout; it must stay consistent with `encode_act`.
- `plan.py` is not reusable for agro (it executes actions in a gym simulator). Multi-step
  rollout evaluation should use `VWorldModel.rollout` directly, not `PlanEvaluator`.

### D. `has_decoder=false` — real behavior

Verified: no decoder module, `visual_pred=None`, `visual_reconstructed=None`, loss = `z_loss`
only, no `eval_images`, no LPIPS, no decode in `openloop_rollout`, checkpoint has no decoder.
Caveat: `save_ckpt` crashes unless `model.train_decoder=False` (audit §6.13); the minimal
latent-only config is `has_decoder=False model.train_decoder=False`.
`metrics.image_metrics` is still imported at `train.py:23` (soft dependency in practice).

### E. ViT-S → ViT-B shape changes

Change only `conf/encoder/dino.yaml::name: "dinov2_vitb14"`.

| Quantity | ViT-S/14 | ViT-B/14 |
|---|---|---|
| `encoder.emb_dim` | 384 | 768 |
| `encoder.patch_size` | 14 | 14 |
| encoder input | 196×196 | 196×196 (unchanged) |
| tokens per frame | 196 | 196 (unchanged) |
| `z` feature dim (concat_dim=1) | 404 | 788 |
| predictor `pos_embedding` | `(1,588,404)` | `(1,588,788)` ✅ |
| predictor `to_qkv.weight` | `(3072,404)` | `(3072,788)` ✅ |
| VQVAE first conv in-channels | 384 | 768 (automatic via `emb_dim=encoder.emb_dim`) |
| VQVAE `channel` config | 384 | **keep 384** — it is conv width, not input dim |
| encoder params | 22.1 M | 86.6 M |

No other shape changes. `num_patches` stays 196, so `num_patches` and the causal mask do not
need editing. Predictor attention inner width stays `heads*dim_head = 16*64 = 1024`;
`to_qkv` grows because its input dim is the token feature dim.

### F. 224-image preprocessing path

1. dataset `get_frames`: THWC uint8 → `/255` → TCHW `[0,1]`;
2. `default_transform(224)` = `Resize(224)`, `CenterCrop(224)`,
   `Normalize([0.5]*3, [0.5]*3)` → `[-1,1]`, shape `(T,3,224,224)`;
3. `VWorldModel.encode_obs`: flatten to `(b*t,3,224,224)`; `Resize(196)`;
4. DINOv2: 196/14 = 14 → 196 patch tokens; **no internal normalization** — the 0.5
   normalization is the only one;
5. decoder reconstructs at 224×224; recon target is the dataset-tensor `obs["visual"]`.

Do not replace 0.5 normalization with ImageNet stats if comparability with upstream is wanted.
For non-square or non-224 inputs, change `Resize(196)` to `Resize((H,W))` and recompute
`num_patches` in `train.py:244-249`; the hard-coded divisor is 16 (`decoder_scale`), not the
encoder patch size.

### G. Predictor swap

`conf/predictor/vit.yaml::_target_` can point at any module with
`__init__(num_patches, num_frames, dim, ...)` and `forward(x:(b,n,d)) -> (b,n,d)`. If the class
does not expose `pos_embedding`, nothing in the trainer cares (only the smoke tools inspect it).
Keep the block-causal structure if actions are to condition the future.

### H. Loss / metrics hooks

All loss logic is inside `VWorldModel.forward` (`:189-271`). There is no separate loss module
and no mask argument. Adding a missing-observation mask or a phenotype-loss head means editing
that method (and the `loss_components` dict consumed by `Trainer.logs_update`). Image metrics
are `metrics/image_metrics.eval_images` (`train.py:498/594`), called only when `has_decoder`.

## 7. Agro-JEPA module → upstream hook

| Agro-JEPA module (later stage) | Upstream class/function to attach to | Notes |
|---|---|---|
| FIP dataset adapter (multi-year, multi-date) | `datasets.traj_dset.TrajDataset` + `TrajSlicerDataset`; `conf/env/fip.yaml::dataset._target_` | implement `get_seq_length/__getitem__`; reuse `get_train_val_sliced` |
| irregular-time handling (Δt) | `TrajSlicerDataset.__getitem__` (window construction) + `VWorldModel.encode_act` / `encode_obs` | no upstream Δt hook; encode Δt into the action channel first, add a token only if ablation requires |
| weather-integral forcing | `action_encoder` (`models/proprio.ProprioceptiveEmbedding` or a replacement), `VWorldModel.encode_act`, `replace_actions_from_z` | action is unsupervised conditioning; width = weather integral width |
| physiological phenotype anchoring | `proprio_encoder`, `VWorldModel.encode_proprio`, `z_proprio_loss` | supervised; add masking in `VWorldModel.forward` for missing dates |
| growth predictor | `conf/predictor/vit.yaml::_target_` → `models.vit.ViTPredictor` replacement | keep `forward(x)->x`; keep causal mask if using forcings |
| multi-step rollout / evaluation | `VWorldModel.rollout` (`:284-309`), `Trainer.openloop_rollout` (`:632`) | `plan.py` is simulator-only; write a latent-rollout evaluator instead |
| training loop / logging | `train.py::Trainer` (`train/val/init_models/init_optimizers/save_ckpt`) | subclass or config-only changes |
| losses | `VWorldModel.forward` | mask-aware latent MSE goes here |
| image-space metrics (optional) | `metrics/image_metrics.eval_images` | only with `has_decoder`; not a scientific metric for FIP |

## 8. What upstream does **not** support (gaps to close in later stages)

1. **Irregular time.** `frameskip` is a uniform integer; no timestamps, no Δt, no variable
   interval. The predictor gets a learned position per slot, so a variable interval must be
   made explicit (action channel or new token).
2. **Missing observations / masks.** No masking anywhere in `forward`; a missing date must be
   dropped during window construction or masked in a custom loss.
3. **Non-RGB / multi-band imagery.** `DinoV2Encoder` uses the hub default `in_chans=3`; no
   channel adaptation. NDVI/NDRE stacks must be reduced to 3 channels or the DINOv2 model must
   be rebuilt with `in_chans≠3` (and pretrained patch weights no longer fit).
4. **No action / passive system.** `Conv1d(0,...)` is invalid; emulate with zeros.
5. **Variable image size.** `Resize(196)` + `num_patches=(img_size//16)^2` are hard-coded.
6. **Semantic/physical units.** `normalize_states` in `preprocessor.py` is planning-only; the
   training loss is in normalized latent space, so phenotype errors need an explicit
   denormalizer to be interpretable.
7. **No environment simulator for agriculture.** `env/` + `planning/` assume gym tasks and
   action execution; they are irrelevant to FIP.

## 9. Override cheat sheet (no file edits)

```bash
# ViT-B
python train.py --config-name train.yaml ... encoder.name=dinov2_vitb14
# latent-only (also avoids the save_ckpt bug)
python train.py --config-name train.yaml ... has_decoder=False model.train_decoder=False
# fewer history frames (must rebuild the predictor; mask globals follow num_hist)
python train.py --config-name train.yaml ... num_hist=2 num_pred=1
# no planning eval during training
python train.py --config-name train.yaml ... plan_settings.plan_cfg_path=null
```
