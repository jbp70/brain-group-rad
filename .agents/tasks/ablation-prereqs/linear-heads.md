# How the linear (rigid / affine) heads receive group information today

Worktree `.worktrees/ablation-prereqs` at main 8e12596. Line numbers refer to that tree.

## Paths of cross-image information into the rigid / affine outputs

| # | Where | Config A | Config B | Code |
|---|---|---|---|---|
| 1 | Encoder stem + L2-L6 group blocks (concat of the group mean before each conv) | yes | no | `models/group_net.py` 90 (`g = cfg.group_config == "A"`), 92 (stem `group=g`), `_conv_block(cfg, f[i], s, g, ...)` for l2-l6 (lines 97-102) |
| 2 | Linear decoder: the four 1x1 "reduction" conv blocks on the 5x7x6 bottleneck | yes | yes | `group_net.py` 139: `_conv_block(cfg, r, 1, True, ...)`, `group=True` hard-coded |
| 3 | Linear decoder: dense blocks. `dense_1` (512) concatenates the group mean; `dense_0` (2048) only with `--dense-group-concat-first` (default False, never used by the final recipe) | yes | yes | `group_net.py` 144: `group=(i > 0 or cfg.dense_group_concat_first)` |
| 4 | Pre-head norm-act, dropout, the two Dense heads | no (LayerNorm per vector) | no | `group_net.py` 149 ff. |
| 5 | GroupCentering of A_raw (barycenter mu, A_i = mu^-1 A_raw_i) | yes | yes | `group_net.py` 373 / 400, `models/transforms.py` 480 |
| 6 | Losses (template / pair terms) | yes | yes | `runtime/losses.py` |
| 7 | BatchNorm in training mode (statistics over the N images of the one group) in the encoder blocks and the reduction blocks | yes | yes | `models/layers.py` 470 (`BatchNormWithSeparateAffine`), `_PreActivation._pre` 636 |

The group mechanism itself is `layers.group_mean_concat` (619): concat(x_i, mean_j x_j) on the channel axis, inside `_PreActivation._pre` after norm and activation. `GroupDenseBlock` (765) does the same on [N, D] vectors.

So in BOTH existing configs the linear heads see the group through the linear decoder (paths 2 and 3) after the bottleneck; config A adds the encoder (path 1). The deform decoder (D5-D1 group blocks, `group_net.py` 235, `group=True` hard-coded) and the SVF centering (`diffeo.center_velocities`, 99) are the deform-side equivalents; the linear tasks never run the deform decoder. Config B ("decoders only") therefore means: linear decoder AND deform decoder grouped.

## Decision (orchestrator, relaying Jay's request "group-blocks only in the encoder"; option 2)

The default policy for `none` holds. For encoder-only the default ("linear heads as in the existing configs") would make the linear branch identical to config A, so the literal variant was chosen instead. Final mapping (a clean 2 x 2 for all three tasks):

| `--group-config` | encoder (stem, L2-L6) | linear decoder (4 reductions + dense_1) | deform decoder (D5-D1) | paper name |
|---|---|---|---|---|
| `none` | no | no | no | no group blocks |
| `E` | yes | no | no | encoder only |
| `B` | no | yes | yes | decoders only (final model) |
| `A` | yes | yes | yes | everywhere (MultiMorph-style) |

In every variant the group stays coupled through GroupCentering (A) / SVF centering (v), the losses, and training-mode BatchNorm (path 7; left as is, it is ablated separately by `--norm instance`; stated in the report). `--dense-group-concat-first true` is rejected for `none` and `E` (it would add a linear-decoder group concat).
