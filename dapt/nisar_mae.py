#!/usr/bin/env python3
"""NISAR GCOV -> two-channel tiles -> MAE ViT-H/14 continued pretraining.

QUICK START (Python 3.10+, PyTorch 2.4+)
  Install a PyTorch build compatible with your GPU/driver using pytorch.org.
  pip install numpy h5py

  # 1. Download the science .h5 files with ASF's authenticated download script.
  # This program reads LOCAL HDF5 files; it does not download Earthdata files.
  python nisar_mae.py inspect /data/one_scene.h5

  # 2. Write scenes.csv with ONE row per local file. Example:
  # path,area,split
  # /data/chad_01.h5,chad,train
  # /data/chad_02.h5,chad,train
  # /data/pantanal_01.h5,pantanal,train
  # /data/po_01.h5,po,val
  # Use your actual filenames. All dates/overlapping acquisitions of an area
  # must have the same split. Keep downstream TEST geography out of this CSV.
  # Optional columns: row_start,row_stop,col_start,col_stop (native pixels,
  # stop exclusive). With empty columns, the ENTIRE scene is eligible.
  # The yellow search box in Vertex does NOT crop the downloaded science file.

  # 3. Extract all qualifying full, nonoverlapping tiles per scene at 20 m.
  python nisar_mae.py prepare --manifest scenes.csv --out tiles

  # 4. Download Meta's official RGB MAE ViT-H checkpoint (several GB):
  # https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_huge.pth
  python nisar_mae.py train --data tiles --checkpoint mae_pretrain_vit_huge.pth \
      --out runs/nisar --batch-size 1 --accum-steps 32 --train-blocks 4

  # Resume the same experiment at an epoch boundary:
  python nisar_mae.py train --data tiles --resume runs/nisar/last.pt \
      --out runs/nisar --batch-size 1 --accum-steps 32 --train-blocks 4

  # Optional selected TRAIN tiles: indices refer to this exact original cache.
  # Omit --selection to train on all tiles with area-balanced sampling.
  python nisar_mae.py train --data tiles --init-sar runs/nisar/best.pt \
      --selection selected_indices.txt --out runs/selected

  # Optional data parallelism (one COMPLETE model per GPU, no memory sharding):
  torchrun --standalone --nproc_per_node=3 nisar_mae.py train --data tiles \
      --checkpoint mae_pretrain_vit_huge.pth --out runs/nisar_ddp \
      --batch-size 1 --accum-steps 16 --train-blocks 4

  # --train-blocks 4: patch embedding + last 4 encoder blocks + encoder norm/
  # CLS token + full decoder train; earlier blocks are frozen. This is a
  # memory-conscious adaptation experiment, NOT full encoder pretraining.
  # --train-blocks -1: unfreeze everything; full AdamW training of ViT-H needs
  # substantial PER-GPU memory. DDP does not pool GPU VRAM. No FSDP here.
  # --amp fp16 for GPUs without bf16 support; --amp off for CPU debugging.
  # Encoder peak LR defaults to 1e-5; the new decoder uses 10x that LR.
  # Refine ONLY the existing decoder from a trained two-channel checkpoint:
  python nisar_mae.py train --data tiles \
      --init-sar runs/nisar_8blocks_60ep/best.pt --decoder-only \
      --out runs/decoder_only --epochs 20 --warmup-epochs 2 \
      --lr 1e-4 --decoder-lr-mult 1 --min-lr 1e-5
  # --init-sar preserves every model weight but starts a new optimizer/schedule.
  # It requires a compatible cache/statistics and does not reinitialize
  # the decoder. --decoder-only freezes ALL encoder parameters, including the
  # input embedding, CLS token and norm; it overrides --train-blocks.
  # Initial validation is saved as initial_validation.json, and best.pt starts
  # with the initial model so refinement cannot silently select a worse model.
  # Resume this NEW run with --resume runs/decoder_only/last.pt --decoder-only
  # and the same optimization settings (omit --init-sar).
  # A small test model exists ONLY for pipeline tests:
  python nisar_mae.py self-test

DATA CONTRACT
  Default science arrays: /science/LSAR/GCOV/grids/frequencyA/{HHHH,HVHV}.
  They are calibrated linear POWER, so dB = 10*log10(power).
  Frequency A/B are sub-bands within LSAR, not L-band versus S-band.
  Two channels retain fixed meanings; missing channels cause an error.
  Integer spatial downsampling averages LINEAR POWER before dB conversion.
  Upsampling/noninteger resampling is rejected. No inter-date registration
  is attempted: this is single-image pretraining, not pair preparation.
  Invalid/nonpositive/explicit-fill pixels are excluded. A coarse pixel is
  valid only if ALL native pixels in its averaging block are valid in both
  channels. Tiles need >=95% valid pixels by default. Missing pixels are
  mean-imputed AFTER normalization and excluded from reconstruction loss.
  This is a numeric validity filter, NOT complete product QA or a layover/
  shadow mask. Review ASF quality reports before admitting scenes.
  dB clipping [-50,10] is an adjustable starting choice, not a physical law.
  Statistics use train tiles only, shared by every split, with no per-image
  min/max normalization and no per-patch target normalization.
  Area-balanced sampling mixes all training areas each epoch.
  MAE hides 75% of spatial tokens, jointly for both channels. Loss measures
  only valid pixels inside hidden tokens. Validation uses fixed token masks.

OUTPUT
  tiles/{index.jsonl,stats.json,scenes.json,tiles/*.npz}
  runs/nisar/{last.pt,best.pt,metrics.jsonl,config.json}
  Checkpoints contain full model, optimizer, scaler, config and preprocessing
  statistics. Best means lowest validation reconstruction loss; assess actual
  usefulness later on geographically held-out labeled change detection.
  The MAE decoder predicts SAR pixels. It does NOT predict change heatmaps.
  To fine-tune, reuse patch_embed/cls_token/pos_embed/blocks/norm and attach a
  paired-image segmentation decoder. Keep the same preprocessing statistics.

DESIGN REFERENCES
  ASF GCOV guide: https://nisar-docs.asf.alaska.edu/gcov/
  Meta architecture/checkpoint: https://github.com/facebookresearch/mae
  Self-contained PyTorch implementation; no timm or cloned MAE repo required.
  Encoder checkpoint parameter names/shapes are checked strictly. RGB input
  kernels are averaged then multiplied by 3/2 and repeated over 2 channels;
  the SAR prediction head is reinitialized. The linked official download is
  encoder-only, so its entire MAE decoder is initialized from scratch here.
  Full MAE checkpoints with a decoder are also supported. This preserves response to an
  identical signal in all input channels, not an RGB-to-SAR physical mapping.
  Respect the original checkpoint's license and terms.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time
from collections import Counter

import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as activation_checkpoint
from torch.utils.data import DataLoader, Dataset, Sampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def scenes_to_csv(args):
    scenes = load_json(args.scenes)
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("Expected a nonempty scenes.json list")
    out = Path(args.out)
    if out.exists():
        raise ValueError("Output exists; choose a new CSV path")
    fields = ["path", "area", "split", "row_start", "row_stop", "col_start", "col_stop"]
    rows = []
    for scene in scenes:
        if not all(k in scene for k in ("path", "area", "split", "window")) or len(scene["window"]) != 4:
            raise ValueError("Each scene needs path, area, split, and a four-value window")
        if scene["split"] not in ("train", "val"):
            raise ValueError("Scene split must be train or val")
        rows.append(dict(zip(fields, [scene["path"], scene["area"], scene["split"], *scene["window"]])))
    with out.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} scene rows to {out}; original windows and splits retained.", flush=True)


def load_json(path):
    return json.loads(Path(path).read_text())


def stable_seed(text, seed):
    return int.from_bytes(hashlib.sha256(f"{seed}:{text}".encode()).digest()[:8], "little")


def grid_path(frequency):
    return f"/science/LSAR/GCOV/grids/frequency{frequency}"


def fill_values(dataset):
    # Respect explicit positive fill values too, without guessing mask codes.
    values = [dataset.fillvalue]
    for name in ("_FillValue", "missing_value"):
        if name in dataset.attrs:
            values.extend(np.asarray(dataset.attrs[name]).reshape(-1).tolist())
    return [float(v) for v in values if v is not None and np.isfinite(v)]


def get_grid(h5, frequency, channels):
    if len(channels) != 2 or len(set(channels)) != 2:
        raise ValueError("Select two distinct diagonal power channels")
    name = grid_path(frequency)
    if name not in h5:
        raise ValueError(f"Missing {name} in {h5.filename}")
    grid = h5[name]
    for ch in channels:
        if ch not in grid:
            raise ValueError(f"Missing {ch} in {h5.filename}; available: {list(grid.keys())}")
        if grid[ch].ndim != 2 or grid[ch].dtype.kind != "f":
            raise ValueError(f"{ch} must be a 2D real floating-point POWER array")
    shape = grid[channels[0]].shape
    if any(grid[ch].shape != shape for ch in channels):
        raise ValueError("Channel shapes do not agree")
    for coord in ("xCoordinates", "yCoordinates", "xCoordinateSpacing", "yCoordinateSpacing", "projection"):
        if coord not in grid:
            raise ValueError(f"Missing grid metadata: {coord}")
    if len(grid["xCoordinates"]) != shape[1] or len(grid["yCoordinates"]) != shape[0]:
        raise ValueError("Coordinate lengths do not agree with science array dimensions")
    return grid


def inspect_file(args):
    with h5py.File(args.file, "r") as h5:
        grid = get_grid(h5, args.frequency, args.channels)
        print("Grid:", grid.name)
        print("Spacing:", float(grid["xCoordinateSpacing"][()]),
              float(grid["yCoordinateSpacing"][()]))
        print("EPSG:", int(grid["projection"][()]))
        for name, ds in grid.items():
            if isinstance(ds, h5py.Dataset):
                print(f"  {name}: shape={ds.shape} dtype={ds.dtype}")
        for ch in args.channels:
            print(f"{ch} explicit/default fill values: {fill_values(grid[ch])}")
    print("Review the product QA report as well; finite power is not sufficient QA.")


def read_manifest(path, require_both=True):
    path = Path(path).resolve()
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or not {"path", "area", "split"}.issubset(rows[0]):
        raise ValueError("CSV needs path,area,split columns and at least one row")
    groups, seen = {}, set()
    for row in rows:
        row.update({k: v.strip() for k, v in row.items() if isinstance(v, str)})
        p = Path(row["path"]).expanduser()
        p = (path.parent / p).resolve() if not p.is_absolute() else p.resolve()
        if not p.is_file():
            raise FileNotFoundError(p)
        if p in seen:
            raise ValueError(f"Duplicate source path: {p}")
        seen.add(p)
        row["path"] = str(p)
        area, split = row["area"], row["split"]
        if not area or split not in ("train", "val"):
            raise ValueError("Each area must be named, and split must be train or val")
        if area in groups and groups[area] != split:
            raise ValueError(f"Geographic group {area!r} crosses train/val")
        groups[area] = split
    if require_both and set(groups.values()) != {"train", "val"}:
        raise ValueError("Provide both training and geographically separate validation areas")
    return rows


def bounds_for(grid, window):
    r0, r1, c0, c1 = window
    dx, dy = abs(float(grid["xCoordinateSpacing"][()])), abs(float(grid["yCoordinateSpacing"][()]))
    x0, x1 = float(grid["xCoordinates"][c0]), float(grid["xCoordinates"][c1-1])
    y0, y1 = float(grid["yCoordinates"][r0]), float(grid["yCoordinates"][r1-1])
    return [min(x0, x1)-dx/2, min(y0, y1)-dy/2,
            max(x0, x1)+dx/2, max(y0, y1)+dy/2]


def check_overlap(scenes):
    # Conservative within-CRS bounding-box check. Full scene bounds include
    # nodata margins. Crop windows if they trigger a false positive. Cross-CRS
    # overlap and a suitable geographic buffer still require manual review.
    for i, a in enumerate(scenes):
        for b in scenes[i+1:]:
            if a["split"] == b["split"] or a["epsg"] != b["epsg"]:
                continue
            x, y = a["bounds"], b["bounds"]
            if max(x[0], y[0]) < min(x[2], y[2]) and max(x[1], y[1]) < min(x[3], y[3]):
                raise ValueError(f"Train/val geographic bounds overlap: {a['path']} and {b['path']}")


def scene_metadata(row, args):
    with h5py.File(row["path"], "r") as h5:
        grid = get_grid(h5, args.frequency, args.channels)
        height, width = grid[args.channels[0]].shape
        r0, r1, c0, c1 = [int(row.get(k) or default) for k, default in
                          (("row_start", 0), ("row_stop", height),
                           ("col_start", 0), ("col_stop", width))]
        if not (0 <= r0 < r1 <= height and 0 <= c0 < c1 <= width):
            raise ValueError(f"Invalid crop window for {row['path']}")
        spacings = [abs(float(grid[k][()])) for k in ("yCoordinateSpacing", "xCoordinateSpacing")]
        factors = []
        for spacing in spacings:
            if not np.isfinite(spacing) or spacing <= 0:
                raise ValueError("Invalid coordinate spacing")
            ratio = args.spacing / spacing
            factor = round(ratio)
            if factor < 1 or not np.isclose(ratio, factor, atol=1e-6, rtol=0):
                raise ValueError(f"Cannot integer-downsample {spacing} m to {args.spacing} m. "
                                 "Use consistent products or externally resample linear power.")
            factors.append(factor)
        epsg = int(grid["projection"][()])
        # UTM or polar stereographic: enforce metre-based projections for this pipeline.
        if not (32601 <= epsg <= 32660 or 32701 <= epsg <= 32760 or epsg in (3413, 3031)):
            raise ValueError(f"Unsupported EPSG:{epsg}; verify projection units before resampling")
        return dict(row, window=[r0, r1, c0, c1], factors=factors,
                    native_spacing=spacings, bounds=bounds_for(grid, [r0, r1, c0, c1]), epsg=epsg)


def read_power_tile(grid, channels, row, col, size, factors, db_min, db_max):
    fy, fx = factors
    height, width = size*fy, size*fx
    arrays, valid_arrays = [], []
    for ch in channels:
        ds = grid[ch]
        power = np.asarray(ds[row:row+height, col:col+width], dtype=np.float32)
        valid = np.isfinite(power) & (power > 0)
        for fill in fill_values(ds):
            valid &= power != fill
        arrays.append(power)
        valid_arrays.append(valid)
    power = np.stack(arrays)
    valid_native = np.logical_and.reduce(valid_arrays)
    power = np.where(valid_native[None], power, 0)
    # Sum/mean in float64 avoids overflow from malformed extreme values.
    power = power.reshape(len(channels), size, fy, size, fx).mean((2, 4), dtype=np.float64)
    valid = valid_native.reshape(size, fy, size, fx).all((1, 3))
    valid &= np.isfinite(power).all(0) & (power > 0).all(0)
    db = 10*np.log10(np.where(valid[None], power, 1))
    clipped = ((db < db_min) | (db > db_max)) & valid[None]
    db = np.clip(db, db_min, db_max).astype(np.float32)
    db[:, ~valid] = 0  # Dataset will replace these with zero AFTER normalization.
    return db, valid, clipped.sum(axis=(1, 2)).astype(np.int64)


def prepare(args):
    if args.tile_size % 14 or args.tile_size <= 0:
        raise ValueError("Tile size must be a positive multiple of ViT-H's patch size 14")
    if not (0.75 <= args.min_valid <= 1) or (args.tiles_per_scene is not None and args.tiles_per_scene < 1) or (args.val_tiles_per_scene is not None and args.val_tiles_per_scene < 1) or args.spacing <= 0:
        raise ValueError("Need min_valid in [0.75,1], positive tile count and spacing")
    if args.db_min >= args.db_max:
        raise ValueError("db-min must be smaller than db-max")
    original_stats = None
    original_fingerprint = None
    if args.normalization_from:
        saved = torch.load(args.normalization_from, map_location="cpu", weights_only=True, mmap=True)
        if not all(k in saved for k in ("model", "stats", "fingerprint", "args")):
            raise ValueError("--normalization-from requires a SAR MAE checkpoint")
        original_stats, original_fingerprint = saved["stats"], saved["fingerprint"]
        expected = dict(version=1, channels=args.channels, frequency=args.frequency,
                        tile_size=args.tile_size, spacing_m=args.spacing,
                        db_clip=[args.db_min, args.db_max], min_valid=args.min_valid)
        mismatch = [k for k, v in expected.items() if k not in original_stats or original_stats[k] != v]
        if mismatch:
            raise ValueError(f"Checkpoint preprocessing differs for {mismatch}; supply matching prepare arguments")
        for key in ("mean", "std"):
            values = np.asarray(original_stats.get(key), dtype=float)
            if values.shape != (2,) or not np.isfinite(values).all() or (key == "std" and (values < 1e-4).any()):
                raise ValueError(f"Invalid checkpoint normalization: {key}")
        del saved
        print("Reusing normalization from the SAR checkpoint; this is a rebuilt cache, not a restored index.", flush=True)
    rows = read_manifest(args.manifest)
    scenes = [scene_metadata(row, args) for row in rows]
    check_overlap(scenes)
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"{out} is not empty; choose a new output directory")
    (out / "tiles").mkdir(parents=True, exist_ok=True)
    write_json(out / "scenes.json", scenes)
    print("Split check passed within each CRS. Manually verify cross-CRS overlap and buffers.", flush=True)
    sums, squares, clip_counts = np.zeros(2), np.zeros(2), np.zeros(2, dtype=np.int64)
    count = 0
    split_counts = Counter()
    with (out / "index.jsonl").open("w") as index:
        for scene_id, scene in enumerate(scenes):
            r0, r1, c0, c1 = scene["window"]
            fy, fx = scene["factors"]
            nr, nc = (r1-r0)//(args.tile_size*fy), (c1-c0)//(args.tile_size*fx)
            # No overlapping crops within one source. Sample grid positions
            # without replacement, then scan until enough valid tiles are found.
            cap = args.tiles_per_scene
            if scene["split"] == "val" and args.val_tiles_per_scene is not None:
                cap = args.val_tiles_per_scene
            if cap is None:
                candidates = range(nr*nc)  # Sequential I/O when exporting the whole grid.
            else:
                rng = np.random.default_rng(stable_seed(scene["path"], args.seed))
                candidates = rng.permutation(nr*nc)
            accepted = 0
            with h5py.File(scene["path"], "r") as h5:
                grid = get_grid(h5, args.frequency, args.channels)
                for position in candidates:
                    r = r0 + (int(position)//nc)*args.tile_size*fy
                    c = c0 + (int(position)%nc)*args.tile_size*fx
                    db, valid, clipped = read_power_tile(grid, args.channels, r, c,
                        args.tile_size, scene["factors"], args.db_min, args.db_max)
                    if valid.mean() < args.min_valid:
                        continue
                    relative = f"tiles/{scene_id:04d}_{r}_{c}.npz"
                    np.savez_compressed(out / relative, db=db, valid=valid)
                    record = dict(file=relative, area=scene["area"], split=scene["split"],
                                  scene_id=scene_id, row=r, col=c, epsg=scene["epsg"],
                                  bounds=bounds_for(grid, [r, r+args.tile_size*fy, c, c+args.tile_size*fx]))
                    index.write(json.dumps(record) + "\n")
                    if scene["split"] == "train":
                        values = db[:, valid].astype(np.float64)
                        sums += values.sum(1)
                        squares += (values**2).sum(1)
                        count += values.shape[1]
                        clip_counts += clipped
                    split_counts[scene["split"]] += 1
                    accepted += 1
                    if accepted % 1000 == 0:
                        print(f"  scene {scene_id+1}: {accepted} tiles accepted", flush=True)
                    if cap is not None and accepted >= cap:
                        break
            print(f"[{scene_id+1}/{len(scenes)}] {scene['area']} {scene['split']}: "
                  f"{accepted} tiles from {Path(scene['path']).name}", flush=True)
            if accepted == 0:
                raise ValueError("No usable tile in this scene. Inspect its footprint, crop and fill values. "
                                 "Incomplete output has no stats.json and cannot be trained.")
    if count == 0 or not all(split_counts[s] > 0 for s in ("train", "val")):
        raise ValueError("Both splits must have usable tiles")
    mean = sums/count
    std = np.sqrt(np.maximum(squares/count-mean**2, 0))
    if np.any(std < 1e-4):
        raise ValueError("Near-constant training channel; inspect inputs before training")
    if original_stats is not None:
        mean = np.asarray(original_stats["mean"], dtype=np.float64)
        std = np.asarray(original_stats["std"], dtype=np.float64)
    stats = dict(version=1, channels=args.channels, frequency=args.frequency,
                 tile_size=args.tile_size, spacing_m=args.spacing,
                 db_clip=[args.db_min, args.db_max], mean=mean.tolist(), std=std.tolist(),
                 train_valid_pixels=count, train_clipped_fraction=(clip_counts/count).tolist(),
                 split_tiles=dict(split_counts), min_valid=args.min_valid, seed=args.seed)
    if original_stats is not None:
        stats["normalization_from_checkpoint_fingerprint"] = original_fingerprint
        stats["normalization_note"] = "mean/std from checkpoint; train_valid_pixels and clipped_fraction describe rebuilt training cache"
    # Written last: acts as the cache-completion marker.
    write_json(out / "stats.json", stats)
    print(json.dumps(stats, indent=2))
    print("Keep source provenance and QA reports. Cached tiles are now sufficient for training.")


def cache_fingerprint(root):
    digest = hashlib.sha256()
    for name in ('index.jsonl', 'stats.json', 'scenes.json'):
        digest.update((Path(root)/name).read_bytes())
    return digest.hexdigest()


def preprocessing_equal(a, b):
    keys = ('version', 'channels', 'frequency', 'tile_size', 'spacing_m',
            'db_clip', 'mean', 'std')
    return all(k in a and k in b and a[k] == b[k] for k in keys)


def compatible_cache(saved, root, allow_appended=False):
    """Accept exact caches, or a verified append ancestor with identical scaling."""
    root = Path(root)
    stats = load_json(root/'stats.json')
    if saved['fingerprint'] == cache_fingerprint(root) and saved['stats'] == stats:
        return True
    if not allow_appended or not preprocessing_equal(saved['stats'], stats):
        return False
    lineage = root/'append_lineage.json'
    if not lineage.is_file():
        return False
    current_index = (root/'index.jsonl').read_bytes()
    current_scenes = load_json(root/'scenes.json')
    for ancestor in load_json(lineage).get('ancestors', []):
        if ancestor['fingerprint'] != saved['fingerprint']:
            continue
        original_stats = ancestor['stats_json'].encode('utf-8')
        original_scenes = ancestor['scenes_json'].encode('utf-8')
        prefix = current_index[:ancestor['index_bytes']]
        digest = hashlib.sha256(prefix + original_stats + original_scenes).hexdigest()
        scenes = json.loads(original_scenes)
        return (digest == saved['fingerprint'] and json.loads(original_stats) == saved['stats']
                and current_scenes[:len(scenes)] == scenes)
    return False


def append_cache(args):
    """Create a new cache; retain the old index byte prefix and raw tile bytes."""
    import shutil
    source, out = Path(args.base).resolve(), Path(args.out).resolve()
    if source == out or source in out.parents or out in source.parents:
        raise ValueError('Base and output must be separate, non-nested directories')
    if out.exists():
        raise ValueError('Append output must not exist; the base cache is never edited')
    stats = load_json(source/'stats.json')
    scenes = load_json(source/'scenes.json')
    raw_index = (source/'index.jsonl').read_bytes()
    if raw_index and not raw_index.endswith(b'\n'):
        raise ValueError('Base index must end in a newline')
    records = [json.loads(line) for line in raw_index.splitlines() if line.strip()]
    if not scenes or not records:
        raise ValueError('Empty base cache')
    for record in records:
        tile = (source/record['file']).resolve()
        if source not in tile.parents or not tile.is_file():
            raise ValueError(f'Invalid/missing base tile: {record["file"]}')
    # Reuse the original physical preprocessing and numeric validity threshold.
    args.frequency, args.channels = stats['frequency'], stats['channels']
    args.tile_size, args.spacing = stats['tile_size'], stats['spacing_m']
    args.db_min, args.db_max = stats['db_clip']
    args.min_valid = stats['min_valid']
    if args.tiles_per_scene < 1:
        raise ValueError('tiles-per-scene must be positive')
    new_scenes = [scene_metadata(row, args) for row in read_manifest(args.manifest, require_both=False)]
    paths = {str(Path(s['path']).resolve()) for s in scenes}
    basenames = {Path(s['path']).name for s in scenes}
    groups = {s['area']: s['split'] for s in scenes}
    for scene in new_scenes:
        if scene['path'] in paths or Path(scene['path']).name in basenames:
            raise ValueError(f'Source scene already present (path or basename): {scene["path"]}')
        if scene['area'] in groups and groups[scene['area']] != scene['split']:
            raise ValueError(f'Area crosses train/val: {scene["area"]}')
        groups[scene['area']] = scene['split']
    check_overlap(scenes + new_scenes)
    print('Same-CRS split overlap check passed. Review cross-CRS geography and buffers manually.', flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=out.name+'.partial-', dir=out.parent))
    try:
        # Real copies avoid modifying base tiles through shared hard links.
        shutil.copytree(source, staging, dirs_exist_ok=True)
        offset = len(scenes)
        if any(not 0 <= r['scene_id'] < offset for r in records):
            raise ValueError('Base scene IDs do not index scenes.json')
        counts = Counter(r['split'] for r in records)
        old_train, old_val = counts['train'], counts['val']
        with (staging/'index.jsonl').open('a') as index:
            for scene_id, scene in enumerate(new_scenes, start=offset):
                r0, r1, c0, c1 = scene['window']
                fy, fx = scene['factors']
                nr, nc = (r1-r0)//(args.tile_size*fy), (c1-c0)//(args.tile_size*fx)
                rng = np.random.default_rng(stable_seed(scene['path'], args.seed))
                accepted = 0
                with h5py.File(scene['path'], 'r') as h5:
                    grid = get_grid(h5, args.frequency, args.channels)
                    for position in rng.permutation(nr*nc):
                        r = r0 + (int(position)//nc)*args.tile_size*fy
                        c = c0 + (int(position)%nc)*args.tile_size*fx
                        db, valid, _ = read_power_tile(grid, args.channels, r, c,
                            args.tile_size, scene['factors'], args.db_min, args.db_max)
                        if valid.mean() < args.min_valid:
                            continue
                        relative = f'tiles/{scene_id:04d}_{r}_{c}.npz'
                        if (staging/relative).exists():
                            raise ValueError(f'Tile filename collision: {relative}')
                        np.savez_compressed(staging/relative, db=db, valid=valid)
                        record = dict(file=relative, area=scene['area'], split=scene['split'],
                            scene_id=scene_id, row=r, col=c, epsg=scene['epsg'],
                            bounds=bounds_for(grid, [r,r+args.tile_size*fy,c,c+args.tile_size*fx]))
                        index.write(json.dumps(record)+'\n')
                        counts[scene['split']] += 1
                        accepted += 1
                        if accepted >= args.tiles_per_scene:
                            break
                if not accepted:
                    raise ValueError(f'No usable tiles: {scene["path"]}')
                print(f'Appended scene {scene_id}: {accepted} {scene["split"]} tiles', flush=True)
        assert (staging/'index.jsonl').read_bytes()[:len(raw_index)] == raw_index
        lineage_path = source/'append_lineage.json'
        ancestors = load_json(lineage_path)['ancestors'] if lineage_path.exists() else []
        ancestors.append(dict(fingerprint=cache_fingerprint(source), index_bytes=len(raw_index),
            stats_json=(source/'stats.json').read_bytes().decode('utf-8'),
            scenes_json=(source/'scenes.json').read_bytes().decode('utf-8')))
        write_json(staging/'append_lineage.json', dict(version=1, ancestors=ancestors))
        write_json(staging/'scenes.json', scenes+new_scenes)
        stats = dict(stats, split_tiles=dict(counts))
        # train_valid_pixels/clipped_fraction continue to describe the original
        # population used for normalization, not a newly computed population.
        stats.setdefault('normalization_source_fingerprint', ancestors[0]['fingerprint'])
        write_json(staging/'stats.json', stats)
        report = dict(old_train_tiles=old_train, old_val_tiles=old_val,
            new_train_tiles=counts['train']-old_train, new_val_tiles=counts['val']-old_val,
            total_train_tiles=counts['train'], total_val_tiles=counts['val'],
            new_train_index_start=old_train, new_train_index_stop_exclusive=counts['train'],
            normalization='Original mean/std retained; pixel counts describe their source population',
            old_index_prefix_preserved=True)
        write_json(staging/'append_report.json', report)
        staging.rename(out)
        print(json.dumps(report, indent=2))
    finally:
        if staging.exists():
            shutil.rmtree(staging)


class Tiles(Dataset):
    def __init__(self, root, split):
        self.root = Path(root)
        self.stats = load_json(self.root / "stats.json")
        records = [json.loads(line) for line in (self.root / "index.jsonl").read_text().splitlines() if line]
        self.records = [r for r in records if r["split"] == split]
        if not self.records:
            raise ValueError(f"No {split} tiles")
        self.mean = np.asarray(self.stats["mean"], np.float32)[:, None, None]
        self.std = np.asarray(self.stats["std"], np.float32)[:, None, None]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        with np.load(self.root / self.records[idx]["file"], allow_pickle=False) as data:
            x = (data["db"].astype(np.float32) - self.mean)/self.std
            valid = data["valid"].astype(bool)
        if not np.isfinite(x[:, valid]).all():
            raise ValueError("Nonfinite value in cached valid pixels")
        x[:, ~valid] = 0
        return torch.from_numpy(x), torch.from_numpy(valid), idx


class AreaSampler(Sampler):
    """Same deterministic global weighted draw, disjoint slices on DDP ranks."""
    def __init__(self, dataset, rank, world, seed):
        counts = Counter(r["area"] for r in dataset.records)
        self.weights = torch.tensor([1/counts[r["area"]] for r in dataset.records], dtype=torch.double)
        self.rank, self.world, self.seed = rank, world, seed
        self.length, self.epoch = math.ceil(len(dataset)/world), 0

    def __len__(self):
        return self.length

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed+self.epoch)
        indices = torch.multinomial(self.weights, self.length*self.world, replacement=True, generator=generator)
        return iter(indices[self.rank::self.world].tolist())


def read_selection(path, dataset):
    import re
    content = Path(path).read_text()
    tokens = [v for v in re.split(r"[,\s]+", content.strip()) if v]
    selected = []
    for token in tokens:
        if re.fullmatch(r"[0-9]+", token):
            selected.append(int(token))
        elif re.fullmatch(r"[0-9]+-[0-9]+", token):
            start, end = map(int, token.split("-"))
            if end < start: raise ValueError(f"Reversed range: {token}")
            selected.extend(range(start, end+1))
        else: raise ValueError(f"Invalid selection token: {token}")
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("Selection must be nonempty and contain no duplicates")
    if min(selected) < 0 or max(selected) >= len(dataset.records):
        raise ValueError(f"Selection indices must be within TRAIN split [0, {len(dataset.records)-1}]")
    return selected


class SelectedSampler(Sampler):
    # A shuffled pass over selected records, with padding for equal DDP steps.
    def __init__(self, dataset, rank, world, seed):
        self.n, self.rank, self.world, self.seed = len(dataset), rank, world, seed
        self.length, self.epoch = math.ceil(self.n/world), 0
    def __len__(self): return self.length
    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.randperm(self.n, generator=generator).tolist()
        padding = self.length*self.world - self.n
        if padding: indices.extend((indices * math.ceil(padding/self.n))[:padding])
        return iter(indices[self.rank::self.world])


def position_embedding(size, dim):
    if dim % 4:
        raise ValueError("Positional dimension must be divisible by 4")
    y, x = torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij")
    omega = 1 / (10000 ** (torch.arange(dim//4, dtype=torch.float64)/(dim//4)))
    vectors = []
    for coordinate in (x, y):
        phase = coordinate.flatten().double()[:, None]*omega[None]
        vectors.extend((phase.sin(), phase.cos()))
    tokens = torch.cat(vectors, dim=1).float()
    return torch.cat((torch.zeros(1, dim), tokens))[None]


class PatchEmbed(nn.Module):
    def __init__(self, dim, patch_size):
        super().__init__()
        self.proj = nn.Conv2d(2, dim, patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, dim*3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        batch, tokens, dim = x.shape
        q, k, v = self.qkv(x).reshape(batch, tokens, 3, self.heads, dim//self.heads).permute(2, 0, 3, 1, 4).unbind(0)
        attended = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        return self.proj(attended.transpose(1, 2).reshape(batch, tokens, dim))


class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(dim, 4*dim), nn.Linear(4*dim, dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim, eps=1e-6), nn.LayerNorm(dim, eps=1e-6)
        self.attn, self.mlp = Attention(dim, heads), MLP(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class SARMAE(nn.Module):
    def __init__(self, image_size=224, arch="huge", grad_checkpoint=True):
        super().__init__()
        dim, depth, heads, dec_dim, dec_depth, dec_heads = {
            "huge": (1280, 32, 16, 512, 8, 16),
            "tiny": (64, 2, 4, 32, 2, 4),  # CPU tests only, not a ViT-H substitute.
        }[arch]
        self.patch_size, self.image_size = 14, image_size
        self.arch, self.grad_checkpoint = arch, grad_checkpoint
        if image_size % self.patch_size:
            raise ValueError("Image dimensions must be divisible by 14")
        self.patch_embed = PatchEmbed(dim, self.patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(position_embedding(image_size//14, dim), requires_grad=False)
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.decoder_embed = nn.Linear(dim, dec_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_dim))
        self.decoder_pos_embed = nn.Parameter(position_embedding(image_size//14, dec_dim), requires_grad=False)
        self.decoder_blocks = nn.ModuleList([Block(dec_dim, dec_heads) for _ in range(dec_depth)])
        self.decoder_norm = nn.LayerNorm(dec_dim, eps=1e-6)
        self.decoder_pred = nn.Linear(dec_dim, 14*14*2)
        # The published Meta encoder checkpoint omits the reconstruction
        # decoder. Initialize its linear layers explicitly before import.
        for module in (self.decoder_embed, self.decoder_blocks, self.decoder_pred):
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.xavier_uniform_(self.decoder_pred.weight)
        nn.init.zeros_(self.decoder_pred.bias)

    def patchify(self, x):
        b, c, h, w = x.shape
        p = self.patch_size
        return x.reshape(b, c, h//p, p, w//p, p).permute(0, 2, 4, 3, 5, 1).reshape(b, (h//p)*(w//p), p*p*c)

    def unpatchify(self, tokens):
        b, length, pixels = tokens.shape
        side, p = math.isqrt(length), self.patch_size
        if side*side != length or pixels != p*p*2:
            raise ValueError("Invalid token shape")
        return tokens.reshape(b, side, side, p, p, 2).permute(0, 5, 1, 3, 2, 4).reshape(b, 2, side*p, side*p)

    def run_blocks(self, blocks, x):
        for block in blocks:
            if self.training and self.grad_checkpoint and torch.is_grad_enabled() and x.requires_grad:
                # Even frozen blocks need a gradient path back to patch_embed.
                x = activation_checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return x

    def forward(self, image, valid, mask_ratio=0.75, noise=None):
        b, c, h, w = image.shape
        if (c, h, w) != (2, self.image_size, self.image_size):
            raise ValueError("Expected [batch,2,image_size,image_size]")
        tokens = self.patch_embed(image) + self.pos_embed[:, 1:]
        n, dim = tokens.shape[1:]
        keep = int(n*(1-mask_ratio))
        if not 0 < keep < n:
            raise ValueError("Mask ratio must retain and hide at least one token")
        if noise is None:
            noise = torch.rand(b, n, device=image.device)
        order = noise.argsort(dim=1)
        restore = order.argsort(dim=1)
        kept = torch.gather(tokens, 1, order[:, :keep, None].expand(-1, -1, dim))
        cls = (self.cls_token+self.pos_embed[:, :1]).expand(b, -1, -1)
        latent = self.norm(self.run_blocks(self.blocks, torch.cat((cls, kept), dim=1)))
        decoded = self.decoder_embed(latent)
        missing = self.mask_token.expand(b, n-keep, -1)
        ordered = torch.cat((decoded[:, 1:], missing), dim=1)
        ordered = torch.gather(ordered, 1, restore[:, :, None].expand(-1, -1, ordered.shape[-1]))
        decoded = torch.cat((decoded[:, :1], ordered), dim=1) + self.decoder_pos_embed
        prediction = self.decoder_pred(self.decoder_norm(self.run_blocks(self.decoder_blocks, decoded)))[:, 1:]
        hidden = torch.ones(b, n, device=image.device)
        hidden[:, :keep] = 0
        hidden = torch.gather(hidden, 1, restore)
        # Accumulate MSE in FP32; invalid pixels never contribute to the loss.
        target = self.patchify(image.float())
        valid_tokens = self.patchify(valid[:, None].expand(-1, 2, -1, -1).float())
        weights = valid_tokens*hidden[:, :, None]
        numerator = ((prediction.float()-target).square()*weights).sum()
        denominator = weights.sum()
        loss = numerator/denominator.clamp_min(1)
        return loss, numerator.detach(), denominator.detach(), prediction, hidden


def load_rgb_checkpoint(model, path):
    # Never silently fall back to unrestricted pickle loading. For old files
    # with extra Python objects, re-export only the model tensor state in a
    # separate trusted environment, then pass that tensor-only file here.
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    state = dict(payload.get("model", payload))
    expected = model.state_dict()
    decoder_keys = {k for k in expected if k.startswith("decoder_") or k == "mask_token"}
    missing_encoder = (set(expected)-decoder_keys)-set(state)
    extra = set(state)-set(expected)
    if missing_encoder or extra:
        raise ValueError(f"Checkpoint encoder mismatch: missing={sorted(missing_encoder)}, extra={sorted(extra)}")
    has_decoder = bool(set(state) & decoder_keys)
    if has_decoder:
        missing_decoder = decoder_keys-set(state)
        if missing_decoder:
            raise ValueError(f"Incomplete decoder checkpoint: missing {sorted(missing_decoder)}")
    else:
        # Only this explicitly defined missing-key set is permitted. Never use
        # unchecked strict=False, which can hide missing encoder weights.
        state.update({k: expected[k] for k in decoder_keys})
    if "patch_embed.proj.weight" not in state:
        raise ValueError("Not an official MAE state dictionary")
    weight = state["patch_embed.proj.weight"]
    if weight.shape[1] != 3:
        raise ValueError("--checkpoint expects RGB MAE weights; use --resume for this script's checkpoints")
    state["patch_embed.proj.weight"] = weight.mean(1, keepdim=True).repeat(1, 2, 1, 1)*1.5
    # No principled correspondence between RGB output pixels and SAR powers.
    state["decoder_pred.weight"] = model.decoder_pred.weight.detach().clone()
    state["decoder_pred.bias"] = model.decoder_pred.bias.detach().clone()
    for key in ("pos_embed", "decoder_pos_embed"):
        old, new = state[key], expected[key]
        if old.shape != new.shape:
            # These are fixed sine/cosine embeddings. Regenerate rather than
            # interpolate when intentionally using a non-224 input size.
            if old.shape[-1] != new.shape[-1]:
                raise ValueError("Checkpoint embedding width does not match architecture")
            state[key] = new
    model.load_state_dict(state, strict=True)
    decoder_status = "loaded decoder; reset SAR output head" if has_decoder else "initialized new MAE decoder"
    print(f"Loaded RGB encoder: adapted input kernel; {decoder_status}. Encoder keys checked strictly.", flush=True)


def load_sar_checkpoint(model, path, stats, fingerprint, cache_root=None, allow_appended=False, allow_rebuilt=False):
    """Weights-only initialization; never import an old optimizer or reset head."""
    saved = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not all(k in saved for k in ("model", "args", "stats", "fingerprint")):
        raise ValueError("--init-sar requires a trained checkpoint from this script")
    exact = saved["stats"] == stats and saved["fingerprint"] == fingerprint
    rebuilt = (allow_rebuilt and preprocessing_equal(saved["stats"], stats)
               and saved["fingerprint"] == stats.get("normalization_from_checkpoint_fingerprint"))
    if not exact and not rebuilt and not (cache_root and compatible_cache(saved, cache_root, allow_appended)):
        raise ValueError("Cache mismatch: use --allow-appended-cache only for a verified append descendant with identical preprocessing")
    if rebuilt and not exact:
        print("Explicit rebuilt-cache initialization: checkpoint normalization matches, but original indices and validation set are not restored.", flush=True)
    if saved["args"]["arch"] != model.arch:
        raise ValueError("--arch must match the SAR checkpoint")
    model.load_state_dict(saved["model"], strict=True)
    print("Loaded full SAR model, including existing decoder. Starting fresh optimizer/schedule.", flush=True)


def set_trainable(model, last_blocks, decoder_only=False):
    if decoder_only:
        for name, param in model.named_parameters():
            param.requires_grad_((name.startswith("decoder_") or name == "mask_token")
                                 and name != "decoder_pos_embed")
        return
    depth = len(model.blocks)
    if last_blocks < -1 or last_blocks > depth:
        raise ValueError(f"train-blocks must be -1 or between 0 and {depth}")
    for name, param in model.named_parameters():
        param.requires_grad_(name not in ("pos_embed", "decoder_pos_embed"))
    if last_blocks >= 0:
        for block in model.blocks[:depth-last_blocks]:
            for param in block.parameters():
                param.requires_grad_(False)


def atomic_save(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def runtime():
    world, rank, local = (int(os.environ.get(k, d)) for k, d in
                          (("WORLD_SIZE", "1"), ("RANK", "0"), ("LOCAL_RANK", "0")))
    if torch.cuda.is_available():
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
    else:
        device = torch.device("cpu")
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    return device, rank, world


def lr_at(step, total, warmup, maximum, minimum):
    if warmup and step < warmup:
        return maximum*(step+1)/warmup
    progress = (step-warmup)/max(1, total-warmup-1)
    return minimum + 0.5*(maximum-minimum)*(1+math.cos(math.pi*min(1, progress)))


@torch.no_grad()
def validate(model, loader, device, amp, ratio, seed, world):
    model.eval()
    sums = torch.zeros(2, dtype=torch.float64, device=device)
    n_tokens = (model.image_size//model.patch_size)**2
    for x, valid, indices in loader:
        # A sample's validation mask is independent of rank and batch size.
        noise = torch.stack([torch.rand(n_tokens, generator=torch.Generator().manual_seed(seed+int(i)))
                             for i in indices]).to(device)
        x, valid = x.to(device, non_blocking=True), valid.to(device, non_blocking=True)
        with amp():
            _, numerator, denominator, _, _ = model(x, valid, ratio, noise)
        sums += torch.stack((numerator, denominator)).double()
    if world > 1:
        dist.all_reduce(sums)
    if sums[1] <= 0:
        raise ValueError("Validation has no valid masked pixels")
    result = (sums[0]/sums[1]).item()
    if not math.isfinite(result):
        raise FloatingPointError("Nonfinite validation loss")
    return result


def train(args):
    if args.allow_appended_cache and not args.init_sar:
        raise ValueError("--allow-appended-cache requires --init-sar; --resume stays strict")
    if args.allow_rebuilt_cache and not args.init_sar:
        raise ValueError("--allow-rebuilt-cache requires --init-sar; --resume stays strict")
    if args.allow_rebuilt_cache and args.allow_appended_cache:
        raise ValueError("Choose one cache compatibility mode")
    if min(args.epochs, args.batch_size, args.accum_steps, args.log_every) < 1 or args.workers < 0:
        raise ValueError("Invalid epoch/batch/accumulation/worker count")
    if not (0 < args.mask_ratio < 1) or not 0 <= args.min_lr <= args.lr or args.lr <= 0:
        raise ValueError("Invalid mask ratio or learning rate")
    if args.clip_grad <= 0 or args.weight_decay < 0 or args.decoder_lr_mult <= 0:
        raise ValueError("clip-grad/decoder-lr-mult must be positive; weight-decay nonnegative")
    if not 0 <= args.warmup_epochs < args.epochs:
        raise ValueError("warmup-epochs must be >=0 and < epochs")
    if args.arch == "huge" and not (args.checkpoint or args.init_sar or args.resume):
        raise ValueError("ViT-H requires --checkpoint, --init-sar or --resume")
    device, rank, world = runtime()
    try:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.amp == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
            raise ValueError("This GPU does not support bf16; select --amp fp16")
        if args.amp == "fp16" and device.type != "cuda":
            raise ValueError("Use --amp off for CPU tests")
        dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
        amp = (lambda: contextlib.nullcontext()) if args.amp == "off" else (
            lambda: torch.autocast(device_type=device.type, dtype=dtype))
        train_data, val_data = Tiles(args.data, "train"), Tiles(args.data, "val")
        stats = train_data.stats
        if len(stats["channels"]) != 2:
            raise ValueError("Exactly two power channels required")
        data_digest = hashlib.sha256()
        for filename in ("index.jsonl", "stats.json", "scenes.json"):
            data_digest.update((Path(args.data)/filename).read_bytes())
        fingerprint = data_digest.hexdigest()
        # No selection preserves the original area-balanced sampling behavior.
        # Selected indices refer to the original TRAIN split, before filtering.
        args.selection_digest = None
        if args.selection:
            selected = read_selection(args.selection, train_data)
            args.selection_digest = hashlib.sha256(
                json.dumps(selected, separators=(",", ":")).encode()
            ).hexdigest()
            original_count = len(train_data.records)
            train_data.records = [train_data.records[i] for i in selected]
            sampler = SelectedSampler(train_data, rank, world, args.seed)
            if rank == 0:
                print(f"Selected {len(selected)} of {original_count} TRAIN tiles from {len(set(r['area'] for r in train_data.records))} areas; DDP epoch padding={sampler.length*world-len(selected)}", flush=True)
                print("Selected scenes:", dict(Counter(str(r.get('scene_id', 'unknown')) for r in train_data.records)), flush=True)
        else:
            sampler = AreaSampler(train_data, rank, world, args.seed)
            if rank == 0:
                print(f"Using all {len(train_data.records)} TRAIN tiles with area-balanced sampling", flush=True)
        loader_kwargs = dict(batch_size=args.batch_size, num_workers=args.workers,
                             pin_memory=device.type == "cuda", persistent_workers=args.workers > 0)
        train_loader = DataLoader(train_data, sampler=sampler, **loader_kwargs)
        val_loader = DataLoader(val_data, sampler=list(range(rank, len(val_data), world)), **loader_kwargs)
        steps_per_epoch = math.ceil(len(train_loader)/args.accum_steps)
        model = SARMAE(stats["tile_size"], args.arch, not args.no_grad_checkpoint)
        resume = None
        if args.resume:
            resume = torch.load(args.resume, map_location="cpu", weights_only=True, mmap=True)
            if resume["args"].get("selection_digest") != args.selection_digest:
                raise ValueError("Resume requires the exact same TRAIN index selection and order")
            if resume["args"].get("decoder_only", False) != args.decoder_only:
                raise ValueError("Resume must preserve --decoder-only; use --init-sar for a new training phase")
            for key in ("arch", "train_blocks", "batch_size", "accum_steps", "epochs", "lr", "min_lr",
                        "warmup_epochs", "weight_decay", "mask_ratio", "amp", "seed", "clip_grad", "decoder_lr_mult"):
                if resume["args"][key] != vars(args)[key]:
                    raise ValueError(f"Resume must preserve {key}; start a new experiment to change it")
            if resume["fingerprint"] != fingerprint or resume["world_size"] != world:
                raise ValueError("Resume data manifest or world size changed")
            model.load_state_dict(resume["model"], strict=True)
        elif args.checkpoint:
            load_rgb_checkpoint(model, args.checkpoint)
        elif args.init_sar:
            load_sar_checkpoint(model, args.init_sar, stats, fingerprint, args.data, args.allow_appended_cache, args.allow_rebuilt_cache)
        set_trainable(model, args.train_blocks, decoder_only=args.decoder_only)
        total = sum(p.numel() for p in model.parameters())
        active = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if rank == 0:
            print(f"Parameters: total={total:,}, trainable={active:,}. Device={device}; world={world}")
            print(f"Effective batch: {args.batch_size*args.accum_steps*world} tiles (last update may be smaller)")
            if args.decoder_only:
                print("DECODER ONLY: all encoder weights frozen; --train-blocks is ignored.")
                print(f"Decoder peak LR = {args.lr*args.decoder_lr_mult:g} (lr * decoder-lr-mult)")
            print("DDP replicates weights/optimizer on each GPU. Full ViT-H AdamW may exceed 16 GB even at batch 1.")
        model.to(device)
        param_groups = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                scale = args.decoder_lr_mult if name.startswith("decoder_") or name == "mask_token" else 1.0
                wd = 0.0 if p.ndim == 1 or name.endswith(".bias") else args.weight_decay
                param_groups.setdefault((scale, wd), []).append(p)
        optimizer = torch.optim.AdamW([
            {"params": params, "weight_decay": wd, "lr_scale": scale, "lr": args.lr*scale}
            for (scale, wd), params in param_groups.items()], betas=(0.9, 0.95), foreach=False)
        scaler = torch.amp.GradScaler("cuda", enabled=args.amp == "fp16")
        start_epoch, step, best = 0, 0, float("inf")
        if resume:
            optimizer.load_state_dict(resume["optimizer"])
            scaler.load_state_dict(resume["scaler"])
            start_epoch, step, best = resume["epoch"]+1, resume["step"], resume["best"]
            del resume
        wrapped = DDP(model, device_ids=[device.index] if device.type == "cuda" else None,
                      broadcast_buffers=False, gradient_as_bucket_view=True) if world > 1 else model
        out = Path(args.out)
        if rank == 0:
            out.mkdir(parents=True, exist_ok=True)
            if not args.resume and any((out/f).exists() for f in ("last.pt", "best.pt", "metrics.jsonl")):
                raise ValueError("Existing run output: use --resume or a new --out directory")
            write_json(out / "config.json", dict(args=vars(args), preprocessing=stats,
                       world_size=world, data_fingerprint=fingerprint))
        if world > 1:
            dist.barrier()
        if args.init_sar:
            # Compare refinement to its actual starting point, under the NEW
            # run's mask ratio/precision and the same fixed validation masks.
            best = validate(model, val_loader, device, amp, args.mask_ratio, args.seed+100000, world)
            if rank == 0:
                write_json(out/"initial_validation.json", dict(val_loss=best,
                    source_checkpoint=str(Path(args.init_sar).resolve()), mask_ratio=args.mask_ratio))
                atomic_save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                    scaler=scaler.state_dict(), epoch=-1, step=0, best=best, args=vars(args),
                    stats=stats, fingerprint=fingerprint, world_size=world), out/"best.pt")
                print(f"Initial validation loss: {best:.8f}; initial model saved as best.pt", flush=True)
            if world > 1:
                dist.barrier()
        for epoch in range(start_epoch, args.epochs):
            t0 = time.monotonic()
            sampler.epoch = epoch
            # Epoch-local RNG makes epoch-boundary resumes reproducible on the
            # same hardware/config. It does not promise cross-device bit identity.
            torch.manual_seed(args.seed + epoch*world + rank)
            wrapped.train()
            if args.decoder_only:
                model.patch_embed.eval()
                model.blocks.eval()
                model.norm.eval()
            optimizer.zero_grad(set_to_none=True)
            totals = torch.zeros(2, dtype=torch.float64, device=device)
            for batch_idx, (x, valid, _) in enumerate(train_loader):
                if batch_idx % args.accum_steps == 0:
                    group_size = min(args.accum_steps, len(train_loader)-batch_idx)
                    lr = lr_at(step, args.epochs*steps_per_epoch,
                               args.warmup_epochs*steps_per_epoch, args.lr, args.min_lr)
                    for group in optimizer.param_groups:
                        group["lr"] = lr*group["lr_scale"]
                update = ((batch_idx+1) % args.accum_steps == 0 or batch_idx+1 == len(train_loader))
                sync = wrapped.no_sync() if world > 1 and not update else contextlib.nullcontext()
                x, valid = x.to(device, non_blocking=True), valid.to(device, non_blocking=True)
                with sync:
                    with amp():
                        loss, numerator, denominator, _, _ = wrapped(x, valid, args.mask_ratio)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite training loss; inspect data / lower LR / use bf16")
                    scaler.scale(loss/group_size).backward()
                totals += torch.stack((numerator, denominator)).double()
                if update:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], args.clip_grad)
                    # FP16's scaler detects overflow and skips the optimizer
                    # update. Without a scaler, abort instead of corrupting weights.
                    if args.amp != "fp16" and not torch.isfinite(grad_norm):
                        raise FloatingPointError("Nonfinite gradient norm; no optimizer update was made")
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    step += 1
                    if rank == 0 and step % args.log_every == 0:
                        print(f"epoch={epoch+1} step={step} lr={lr:.3g} batch_loss={loss.item():.5f}", flush=True)
            if world > 1:
                dist.all_reduce(totals)
            train_loss = (totals[0]/totals[1]).item()
            # Use raw model: validation ranks may have unequal batch counts.
            val_loss = validate(model, val_loader, device, amp, args.mask_ratio, args.seed+100000, world)
            improved = val_loss < best
            best = min(best, val_loss)
            if rank == 0:
                record = dict(epoch=epoch+1, step=step, train_loss=train_loss, val_loss=val_loss,
                              best=best, lr=lr, seconds=time.monotonic()-t0)
                with (out/"metrics.jsonl").open("a") as stream:
                    stream.write(json.dumps(record, allow_nan=False)+"\n")
                snapshot = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                                epoch=epoch, step=step, best=best, args=vars(args), stats=stats,
                                fingerprint=fingerprint, world_size=world)
                atomic_save(snapshot, out/"last.pt")
                if improved:
                    atomic_save(snapshot, out/"best.pt")
                print(json.dumps(record), flush=True)
            if world > 1:
                dist.barrier()
        if rank == 0:
            print(f"Complete. Best MAE checkpoint: {out/'best.pt'}")
    finally:
        if world > 1 and dist.is_initialized():
            dist.destroy_process_group()


def self_test(_args=None):
    """Synthetic CPU checks; no real science product/checkpoint needed."""
    torch.set_num_threads(2)
    torch.manual_seed(7)
    model = SARMAE(28, "tiny", True)
    x = torch.randn(2, 2, 28, 28)
    valid = torch.ones(2, 28, 28, dtype=torch.bool)
    valid[:, :2, :2] = False
    x[:, :, :2, :2] = 0
    assert torch.equal(model.unpatchify(model.patchify(x)), x)
    loss, _, denom, pred, hidden = model(x, valid)
    assert hidden.sum().item() == 6 and denom > 0 and torch.isfinite(loss)
    assert pred.shape == (2, 4, 392)
    loss.backward()
    assert model.patch_embed.proj.weight.grad.abs().sum() > 0
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)/"synthetic.h5"
        with h5py.File(path, "w") as h5:
            grid = h5.create_group(grid_path("A"))
            # Known powers: a coarse block averages (1,1,9,9) to 5 -> 6.9897 dB.
            array = np.ones((56, 56), np.float32)
            array[1::2] = 9
            for ch in ("HHHH", "HVHV"):
                ds = grid.create_dataset(ch, data=array)
                ds.attrs["_FillValue"] = 9999.0
            grid["HHHH"][0, 0] = 9999.0
            db, good, _ = read_power_tile(grid, ["HHHH", "HVHV"], 0, 0, 28, [2, 2], -50, 10)
            assert not good[0, 0] and good[0, 1]
            assert np.allclose(db[:, 0, 1], 10*np.log10(5))
        # Import adapter: strict state loading using RGB-shaped synthetic weights.
        rgb = {k: v.detach().clone() for k, v in model.state_dict().items()}
        rgb["patch_embed.proj.weight"] = torch.randn(64, 3, 14, 14)
        rgb["decoder_pred.weight"] = torch.randn(588, 32)
        rgb["decoder_pred.bias"] = torch.randn(588)
        ckpt = Path(tmp)/"rgb.pt"
        torch.save({"model": rgb}, ckpt)
        loaded = SARMAE(28, "tiny")
        load_rgb_checkpoint(loaded, ckpt)
        assert torch.allclose(loaded.patch_embed.proj.weight[:, 0], rgb["patch_embed.proj.weight"].mean(1)*1.5)
        # The actual public Meta ViT-H download is encoder-only.
        encoder = {k: v for k, v in rgb.items() if not k.startswith("decoder_") and k != "mask_token"}
        torch.save({"model": encoder}, ckpt)
        load_rgb_checkpoint(loaded, ckpt)
    print("PASS: two-channel patch round-trip, masking, backward pass, power averaging, fill exclusion, RGB adapter.")


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = p.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect", help="Inspect one local HDF5 science file")
    inspect.add_argument("file")
    catalog = commands.add_parser("scenes-to-csv", help="Create a manifest from a retained scenes.json")
    catalog.add_argument("--scenes", required=True)
    catalog.add_argument("--out", required=True)
    prep = commands.add_parser("prepare", help="Window-read local HDF5s and create a tile cache")
    for sub in (inspect, prep):
        sub.add_argument("--frequency", choices=["A", "B"], default="A")
        sub.add_argument("--channels", nargs=2, default=["HHHH", "HVHV"],
                         choices=["HHHH", "HVHV", "VVVV", "VHVH", "RHRH", "RVRV"])
    prep.add_argument("--manifest", required=True)
    prep.add_argument("--out", required=True)
    prep.add_argument("--tile-size", type=int, default=224)
    prep.add_argument("--tiles-per-scene", type=int, default=None, help="Maximum accepted tiles per scene; omit for all full valid grid tiles")
    prep.add_argument("--val-tiles-per-scene", type=int, default=None, help="Optional separate validation tile cap")
    prep.add_argument("--normalization-from", help="Reuse mean/std from this SAR checkpoint after checking prepare settings; enables explicit rebuilt-cache warm start")
    prep.add_argument("--spacing", type=float, default=20.0)
    prep.add_argument("--min-valid", type=float, default=0.95)
    prep.add_argument("--db-min", type=float, default=-50)
    prep.add_argument("--db-max", type=float, default=10)
    prep.add_argument("--seed", type=int, default=42)
    app = commands.add_parser("append", help="Append new scenes to a NEW cache, preserving old tile indices")
    app.add_argument("--base", required=True, help="Existing completed tile cache")
    app.add_argument("--manifest", required=True, help="CSV containing NEW scenes only; may be train-only")
    app.add_argument("--out", required=True, help="New nonexisting cache directory")
    app.add_argument("--tiles-per-scene", type=int, default=256)
    app.add_argument("--seed", type=int, default=42)
    t = commands.add_parser("train", help="MAE continued pretraining; no t0/t1 labels required")
    t.add_argument("--data", required=True)
    t.add_argument("--allow-appended-cache", action="store_true", help="Allow --init-sar from a verified ancestor cache; retain normalization; reset optimizer")
    t.add_argument("--allow-rebuilt-cache", action="store_true", help="Allow --init-sar for a fresh cache explicitly built with matching checkpoint normalization; not an append")
    t.add_argument("--selection", help="Optional TRAIN split indices and inclusive ranges; omit for area-balanced sampling")
    t.add_argument("--out", required=True)
    group = t.add_mutually_exclusive_group()
    group.add_argument("--checkpoint", help="Official RGB MAE .pth, or tensor-only state dictionary")
    group.add_argument("--init-sar", help="Trained two-channel checkpoint: keep all weights, reset optimizer/schedule; compatible cache required")
    group.add_argument("--resume", help="last.pt produced by this script")
    t.add_argument("--arch", choices=["huge", "tiny"], default="huge")
    t.add_argument("--train-blocks", type=int, default=4)
    t.add_argument("--decoder-only", action="store_true", help="Freeze entire encoder; train reconstruction decoder only; overrides --train-blocks")
    t.add_argument("--batch-size", type=int, default=1)
    t.add_argument("--accum-steps", type=int, default=32)
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--warmup-epochs", type=int, default=3)
    t.add_argument("--lr", type=float, default=1e-5, help="Absolute peak LR, not batch-scaled")
    t.add_argument("--decoder-lr-mult", type=float, default=10.0, help="LR multiplier for MAE reconstruction decoder")
    t.add_argument("--min-lr", type=float, default=1e-6)
    t.add_argument("--weight-decay", type=float, default=0.05)
    t.add_argument("--clip-grad", type=float, default=1.0)
    t.add_argument("--mask-ratio", type=float, default=0.75)
    t.add_argument("--workers", type=int, default=4)
    t.add_argument("--amp", choices=["bf16", "fp16", "off"], default="bf16")
    t.add_argument("--no-grad-checkpoint", action="store_true")
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--log-every", type=int, default=10)
    commands.add_parser("self-test", help="Run small synthetic CPU checks")
    return p


def main():
    args = parser().parse_args()
    {"inspect": inspect_file, "scenes-to-csv": scenes_to_csv, "prepare": prepare, "append": append_cache, "train": train, "self-test": self_test}[args.command](args)


if __name__ == "__main__":
    main()
