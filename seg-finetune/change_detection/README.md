# NISAR full-scene change fine-tuning

This bundle streams your existing paired NISAR GCOV HDF5 scenes into a separate, appendable change-detection dataset, generates draft labels with your existing `nisar_auto_labels.py`, and fine-tunes your SAR MAE encoder with a new dense change head.

**Labels mean candidate SAR backscatter change.** They are not verified flooding, forest loss, or land-cover labels. Training on them is weak supervision: validation against the same heuristic measures agreement with that heuristic. Review held-out examples before making claims about real change-detection accuracy.

## Included files

| File | Purpose |
|---|---|
| `build_change_dataset.py` | Read whole scene pairs in windows; create paired GeoTIFF tiles, draft labels, persistent indices, manifests |
| `finetune_change.py` | Single-GPU / multi-GPU fine-tuning, evaluation, epoch-boundary resume |
| `change_model.py` | Strictly compatible SARMAE encoder and new dense change head |
| `predict_change.py` | Predict all requested tiles, including tiles without usable pseudo-labels |
| `stitch_nisar_labels.py` | Your existing label/prediction and T0/T1 mosaic utility |
| `nisar_auto_labels.py` | Your existing label rules and manual-correction utility, unchanged |
| `nisar_label_pairs.py` | Your existing HDF5/grid and GeoTIFF helpers, unchanged |
| `test_pipeline.py` | Synthetic data, training, loading, and append regression tests |

No `timm` or original `nisar_mae.py` import is required. The encoder matches the earlier SARMAE source found with your project: ViT-H/14, 32 blocks, width 1280, 16 attention heads, two-channel input. Loading checks **all encoder keys and shapes strictly**; an incompatible checkpoint fails instead of silently using random encoder weights. Only reconstruction-decoder keys and the mask token are discarded.

## 1. Install and choose paths

Extract this folder, activate your existing `.venv-nisar`, and run commands from this folder. Keep the PyTorch build that already works with your GPUs.

```bash
pip install -r requirements-data.txt
```

The scripts require Python 3.10+ and PyTorch 2.4+ in addition to these data packages. Use a new dataset directory, not your pretraining `tiles` cache. Your existing `scene_pairs.csv` is accepted:

```csv
pair_id,area,split,t0_path,t1_path
example_pair,example_area,train,/absolute/path/t0.h5,/absolute/path/t1.h5
```

Extra existing columns (dates, track, frame, etc.) are retained in provenance. Relative source paths resolve against the scene CSV's directory.

## 2. Plan whole-scene extraction

This example reserves **Nepal for validation and Netherlands for testing**, using the remaining locations for training. These are explicit example choices, not an assertion that these are the optimal held-out locations. If you already filled `split` in the CSV, omit the two area flags. A conflict between CSV splits and flags is rejected.

```bash
python build_change_dataset.py \
  --manifest /mnt/linux-data/post_train/scene_pairs.csv \
  --stats /mnt/linux-data/pretrain/tiles \
  --out /mnt/linux-data/post_train/change_dataset \
  --val-areas nepal \
  --dry-run
```

Use the exact lowercase area strings in your CSV. `--stats` must point to the preprocessing statistics associated with your chosen MAE checkpoint: channel order, dB clipping, spacing, mean, and standard deviation are preserved. Neither validation nor test data are used to recompute normalization.

The dry run prints the complete grid, maximum tile count and an **uncompressed raster payload estimate**. It does not create tiles or labels. Actual storage also includes filesystem overhead; compression varies with data. Whole-scene extraction can produce thousands of tiles per pair, substantially more than 128 per scene.

By default, tile width and height equal `stats.json`'s `tile_size`, usually 224. A large export crop such as 896 is not necessary for full coverage. This fine-tuner requires the MAE's original tile size, avoiding a hidden change in token count or positional embeddings.

## 3. Generate the dataset and draft labels

Run the same command without `--dry-run`:

```bash
python build_change_dataset.py \
  --manifest /mnt/linux-data/post_train/scene_pairs.csv \
  --stats /mnt/linux-data/pretrain/tiles \
  --out /mnt/linux-data/post_train/change_dataset \
  --val-areas nepal \
  --test-areas netherlands
```

To start with one **whole** pair, add `--pair-id amazon_147_A_001_2005_A`. Remove that option when ready to append the rest. A one-pair training-only trial cannot provide the validation required by the trainer.

What it does:

- Covers the complete **common T0/T1 footprint**, using a regular grid with no random crop selection or crop limit.
- Reads HDF5 windows rather than loading the whole scene into RAM. Native linear powers are averaged to the original pretraining spacing before dB conversion.
- Pads the final tiles; incomplete averaging cells and pixels without valid data at both dates are excluded. Only tiles with **zero jointly valid pixels** are skipped. Some exported tiles may have no confident labels; those remain available for prediction but are excluded from supervised training.
- Labels a larger window, including a 32-pixel halo, then saves its central tile. This reduces smoothing artifacts at tile borders. Small-component filtering is still local, so this is not guaranteed identical to running connected components once over an entire scene.
- Writes `labels_auto.tif` and a separate `labels_editable.tif`. Training uses the editable copy.
- Assigns stable sequential `index` values and commits a record after every completed tile and checkpoints the CSV every 250 tiles and after each pair. Rerunning with the same settings resumes extraction. Existing labels and manually corrected files are not overwritten.

Draft label values:

| Value | Meaning | Used in loss? |
|---|---|---|
| 0 | Confidently small backscatter difference under the heuristic | Yes |
| 1 | Candidate change consistent across both channels and two smoothing scales | Yes |
| 255 | Uncertain, invalid, clipped, or boundary pixel | No |

Default rules are your existing ones: `--low-db 1 --high-db 3 --window 5 --min-pixels 16`. These are heuristics, not universal physical thresholds. A scene-wide moisture/calibration shift can still produce widespread positive labels, and changes affecting only one channel may be missed.

Output:

```text
change_dataset/
  stats.json
  build_config.json
  pairs.json
  dataset_manifest.csv
  build_summary.json
  crops/<pair_id>/<crop_id>/
    t0_db.tif
    t1_db.tif
    valid.tif
    labels_auto.tif
    labels_editable.tif
    record.json
```

CSV paths are relative to the dataset directory, so moving the complete dataset preserves them. Source HDF5 paths and size/mtime metadata are recorded for extraction resume. Training does not need the HDF5 files once extraction has finished. Move or modify source files only when not resuming extraction; changed sources are rejected.

### Appending more scene pairs

Run the builder against the same `--out`, with an expanded scene manifest or a new manifest of additional pairs, keeping preprocessing and labeling settings identical. Existing tile indices stay fixed and new indices are appended. For a new manifest, use its explicit `split` column if your original held-out area names are absent. Previously stored pairs participate in overlap checks too.

This appends to the **new change dataset**, not your pretraining tile cache. It does not import or renumber your old selected pretraining indices.

### Geometry and splits

The builder verifies CRS, native spacing and integer-pixel grid offsets. It rejects unsupported/fractional-grid alignment instead of silently resampling. Equal grids do **not** prove physical sub-pixel coregistration; inspect stable features in T0/T1 before trusting labels.

Whole areas and any overlapping geographic bounds must share a split. Bounding-box checks are conservative and may group scenes that only overlap in no-data margins. No bypass is provided. All dates of the same place must stay together. The scene intersection, not the union, is eligible for temporal supervision.

Also audit your earlier MAE pretraining coverage: a downstream held-out area previously seen during pretraining is not an entirely unseen-geography test. The builder cannot infer your old pretraining footprints.

## 4. Review labels before training

Open T0, T1 and `labels_editable.tif` in QGIS. Use identical channel and display ranges for the two dates. The exported labels include a color table; class 1 is red and 255 is NoData. You can make class 0 transparent for an overlay.

Inspect at least representative changed, unchanged and ambiguous tiles from each area. Preserve uncertain regions as 255. Fix large obvious false-positive regions before a long run. To use your existing polygon corrections:

```bash
python nisar_auto_labels.py correct \
  --label /absolute/path/to/labels_editable.tif \
  --valid /absolute/path/to/valid.tif \
  --polygons /absolute/path/to/corrections.geojson \
  --epsg 32722 \
  --out /absolute/path/to/labels_corrected.tif
```

Replace the example EPSG with the raster's actual CRS. Update the matching CSV row's `label` path to the corrected raster. Keep `label_status=pseudo_unreviewed` or use `partially_reviewed` unless **all labeled pixels in that tile have been checked**. Use `reviewed` only for fully reviewed tiles. Unchecked areas can be set to 255.

The trainer reads the raster values anew; manifest count fields are extraction-time summaries and are not treated as truth after editing.

## 5. First run: train the new head, freeze the encoder

Set the checkpoint path to the MAE you actually want to use. The example below uses a checkpoint shown earlier in your project; replace it if a newer checkpoint is preferred.

```bash
CUDA_VISIBLE_DEVICES=0,1,2 torchrun \
  --standalone --nproc_per_node=3 \
  finetune_change.py \
  --data /mnt/linux-data/post_train/change_dataset \
  --init-mae /mnt/linux-data/pretrain/runs/nisar_expanded_60ep/best.pt \
  --out /mnt/linux-data/post_train/runs/change_head_5ep \
  --train-blocks 0 \
  --epochs 5 --warmup-epochs 1 \
  --batch-size 1 --accum-steps 8 \
  --lr 1e-4 \
  --amp bf16
```

This is supervised/weakly supervised segmentation, **not MAE reconstruction**. Both dates are fed without masking. There is no `--mask-ratio`.

Architecture:

1. The same pretrained encoder processes T0 and T1 separately, keeping all spatial patch tokens.
2. Mean features and absolute feature differences feed a new convolutional fusion head.
3. Features are upsampled to pixel resolution and combined with a small shallow branch from the mean/difference of the input images.
4. A single output logit per pixel predicts binary change.

The shallow branch supplies local detail absent from the coarse token grid. It may also learn the labeling heuristic directly; compare against that heuristic and test on reviewed labels before claiming the encoder adds value. The output is symmetric in T0/T1; it detects change magnitude, not change direction or cause. There is no cross-attention in this first baseline.

The original reconstruction decoder is dropped. `--train-blocks 0` freezes **all** encoder parameters, including patch embedding and normalization. The new head trains with pixel-weighted BCE; ignored pixels contribute no loss. Default positive-class weight is 1. A different `--pos-weight` can be tried in a separate run if class imbalance warrants it, not selected on test data.

The encoder is instantiated from known compatible code, not guessed from the checkpoint size. The loader checks model keys and shapes and compares preprocessing. If it rejects your latest checkpoint, send the exact error and current `nisar_mae.py`; do not weaken strict loading.

## 6. Optional second phase: fine-tune the last four blocks

After evaluating the frozen-encoder run:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 torchrun \
  --standalone --nproc_per_node=3 \
  finetune_change.py \
  --data /mnt/linux-data/post_train/change_dataset \
  --init-change /mnt/linux-data/post_train/runs/change_head_5ep/best.pt \
  --out /mnt/linux-data/post_train/runs/change_last4_15ep \
  --train-blocks 4 \
  --epochs 15 --warmup-epochs 1 \
  --batch-size 1 --accum-steps 8 \
  --lr 3e-5 --encoder-lr 1e-6 \
  --amp bf16
```

This keeps the learned change head, unfreezes the last four encoder blocks plus final encoder norm, and starts a fresh optimizer/schedule. Patch embedding, CLS token and earlier blocks remain frozen. Gradient checkpointing is enabled in trained encoder blocks. DDP replicates the model on each GPU; it does not combine GPU memory. Two unmasked dates may use more compute/memory than your earlier masked reconstruction run.

On one GPU, use `CUDA_VISIBLE_DEVICES=0 python finetune_change.py ...`. Start with batch size 1; increase only after observing peak memory.

## 7. Checkpoints, evaluation, and resume

Each completed epoch saves `last.pt`; lower validation BCE saves `best.pt`. A starting checkpoint is also evaluated and saved as the initial best. If fully reviewed validation tiles exist, **reviewed validation BCE** controls best selection. Otherwise it uses validation BCE against pseudo-labels and says so explicitly.

Metrics are reported separately for all, reviewed, and pseudo-labeled pixels: BCE, change IoU, F1, precision, recall, and confusion counts. Metrics use threshold 0.5 and ignore 255. Undefined ratios are `null`. A good pseudo-label IoU alone is not evidence of scientific correctness or calibrated confidence. Scores are pixel-micro averages and can be dominated by large/easy areas; inspect results by region and review difficult cases.

True resume uses `--resume .../last.pt` **in place of** `--init-mae` or `--init-change`, keeping all other training options, total epoch target, dataset, and GPU count the same. Optimizer/scaler/schedule state is restored at the last completed epoch. Interrupting midway through an epoch loses that epoch's progress. Label edits or dataset additions require a new phase with `--init-change` and a new output directory.

Test is excluded from training and model selection. Evaluate it once after choosing the model:

```bash
python finetune_change.py \
  --data /mnt/linux-data/post_train/change_dataset \
  --init-change /mnt/linux-data/post_train/runs/change_last4_15ep/best.pt \
  --out /mnt/linux-data/post_train/runs/change_test_evaluation \
  --eval-only --eval-split test --amp bf16
```

Automatic labels in test still measure heuristic agreement. Review an independent test subset for real assessment. Repeated tuning against test would turn it into validation.

## 8. Predict a complete pair and reconstruct it

```bash
python predict_change.py \
  --data /mnt/linux-data/post_train/change_dataset \
  --checkpoint /mnt/linux-data/post_train/runs/change_last4_15ep/best.pt \
  --pair-id amazon_147_A_001_2005_A \
  --split all \
  --out /mnt/linux-data/post_train/predictions_amazon \
  --amp bf16

python stitch_nisar_labels.py \
  --manifest /mnt/linux-data/post_train/predictions_amazon/prediction_manifest.csv \
  --out /mnt/linux-data/post_train/predictions_amazon_mosaic
```

Each tile has a float sigmoid-score GeoTIFF and a thresholded mask. The stitcher mosaics binary masks plus T0/T1, not probability rasters. The mask mosaic is named `labels_stitched.tif` for compatibility even though these are model predictions. Non-overlapping inference tiles may exhibit model-context seams; overlapping inference with score blending is a future refinement, not implemented here. Pixels outside joint valid coverage stay NoData.

To reconstruct the draft labels instead, give the stitcher `change_dataset/dataset_manifest.csv` and optionally `--pair-id ...`.

## Verification and limits

```bash
OMP_NUM_THREADS=1 python test_pipeline.py
```

Synthetic tests cover full grid coverage and padded borders, correct linear-power averaging, append/resume and preservation of corrections, strict encoder loading, temporal symmetry, ignored-pixel gradients, frozen/partially unfrozen encoder gradients, two-epoch CPU training, checkpoint reload, prediction export, and rejection of grid/split conflicts. Set `NISAR_TEST_DDP=1` to enable the optional two-rank CPU comparison test (local port 29591 must be free). This environment blocked Gloo socket creation, so that distributed test could not be completed here; single-process tests passed. CUDA/NCCL remains untested.

The encoder was also checked numerically against the earlier SARMAE implementation using its tiny test architecture and shared weights. No real HDF5 scenes, trained 657M-parameter checkpoint, or your GPUs are available in this environment, so real-scene and CUDA training validation still need your local run. Full-scene labels retain the limitations of the existing threshold-based labeler; no automatic physical registration or semantic event labeling is added.

Implementation references: [PyTorch DDP](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html), [Rasterio windowed I/O](https://rasterio.readthedocs.io/en/stable/topics/windowed-rw.html).
