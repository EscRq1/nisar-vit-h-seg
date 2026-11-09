#!/usr/bin/env python3
"""Make a full-scene change mask from an aligned pair of HH/HV SAR PNGs."""
import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, PngImagePlugin
import torch

from change_model import ChangeModel, load_payload

ENCODING_KEY = "nisar_sar_encoding_v1"
TILE_SIZE = 224


def ranges(value):
    value = np.asarray(value, dtype=np.float32)
    if value.shape != (2, 2) or not np.isfinite(value).all() or (value[:, 1] <= value[:, 0]).any():
        raise ValueError("Expected finite, increasing dB ranges for HH and HV.")
    return value


def checkpoint_spec(path):
    payload = load_payload(path)
    if payload.get("format") != "nisar_change_v1" or payload.get("size") != TILE_SIZE:
        raise ValueError("Expected a fine-tuned nisar_change_v1 checkpoint for 224×224 tiles.")
    stats = payload["stats"]
    if tuple(stats["channels"]) != ("HHHH", "HVHV"):
        raise ValueError("Checkpoint must use HH, HV in that order.")
    clip = ranges([stats["db_clip"]] * 2)
    return payload, clip


def metadata(im, path, assume_rg, supplied, clip):
    if im.format != "PNG" or im.mode not in ("RGB", "RGBA"):
        raise ValueError(f"{path}: use an RGB/RGBA SAR PNG (red = HH, green = HV).")
    entry = im.info.get(ENCODING_KEY)
    if entry is not None:
        if supplied is not None:
            raise ValueError(f"{path}: tagged input already records its dB ranges; omit --hh-db/--hv-db.")
        data = json.loads(entry)
        if data.get("channel_order") != ["HH", "HV"]:
            raise ValueError(f"{path}: unsupported PNG channel order.")
        return ranges(data["db_ranges"]), data.get("grid")
    if not assume_rg:
        raise ValueError(f"{path}: no SAR encoding metadata. Use 'pack', or pass --assume-rg "
                         "if you know red and green contain linearly encoded HH and HV dB.")
    return (supplied if supplied is not None else clip), None


def read_tile(im, box, db_ranges):
    cropped = im.crop(box)
    # RGBA has explicit validity; RGB can also carry a transparent color in PNG metadata.
    if cropped.mode == "RGB" and "transparency" in im.info:
        cropped = cropped.convert("RGBA")
    pixels = np.asarray(cropped)
    valid = pixels[..., 3] > 0 if pixels.shape[-1] == 4 else np.ones(pixels.shape[:2], dtype=bool)
    low, high = db_ranges.T[:, :, None, None]
    db = (pixels[..., :2].transpose(2, 0, 1).astype(np.float32) / 255.0) * (high - low) + low
    return db, valid


def offsets(length, stride):
    # Keep the fixed grid and pad only the trailing tile when needed.
    return range(0, max(0, length - TILE_SIZE) + stride, stride)


def score_alpha(scores):
    alpha = np.rint(np.clip(scores, 0, 1) * 255).astype(np.uint8)
    alpha[(scores > 0) & (alpha == 0)] = 1
    return alpha


def overlay_png(alpha, blue=(0, 0, 255)):
    # Each palette index also represents its opacity. Index zero is transparent red.
    im = Image.fromarray(alpha).convert("P")
    im.putpalette([255, 0, 0] + list(blue) * 255)
    im.info["transparency"] = bytes(range(256))
    return im


def infer_scene(model, images, encodings, mean, std, clip, device, amp, batch_size, overlap,
                threshold, save_scores, save_overlay=False, save_mask=True):
    width, height = images[0].size
    stride = TILE_SIZE - overlap
    rows, cols = offsets(height, stride), offsets(width, stride)
    total = len(rows) * len(cols)
    print(f"Detector: {width}×{height} pixels, {total:,} tiles of {TILE_SIZE}×{TILE_SIZE}, "
          f"overlap {overlap}, {device}, AMP {amp}.", flush=True)
    # Rolling strip: float accumulators scale with width × tile size, not scene area.
    sums = np.zeros((TILE_SIZE, width), np.float32)
    weights = np.zeros_like(sums)
    output = np.zeros((height, width, 2), np.uint8) if save_mask else None
    score_png = np.zeros((height, width), np.uint16) if save_scores else None
    alpha = np.zeros((height, width), np.uint8) if save_overlay else None
    taper = np.maximum(np.hanning(TILE_SIZE).astype(np.float32), .05) if overlap else np.ones(TILE_SIZE, np.float32)
    kernel = taper[:, None] * taper[None, :]
    amp_dtype = torch.bfloat16 if amp == "bf16" else torch.float16
    valid_count = changed_count = done = 0
    with torch.inference_mode():
        for row in rows:
            h = min(height - row, TILE_SIZE)
            for start in range(0, len(cols), batch_size):
                inputs = [[], []]
                places = []
                for col in cols[start:start + batch_size]:
                    w = min(width - col, TILE_SIZE)
                    box = (col, row, col + w, row + h)
                    x, y = [read_tile(im, box, enc) for im, enc in zip(images, encodings, strict=True)]
                    valid = x[1] & y[1]
                    if not valid.any():
                        continue
                    for (db, _), items in zip((x, y), inputs, strict=True):
                        tile = np.zeros((2, TILE_SIZE, TILE_SIZE), np.float32)
                        db = np.clip(db, clip[:, 0, None, None], clip[:, 1, None, None])
                        tile[:, :h, :w] = np.where(valid[None], (db - mean) / std, 0.0)
                        items.append(tile)
                    places.append((col, w, valid))
                if places:
                    tensors = [torch.from_numpy(np.stack(items)).to(device) for items in inputs]
                    with torch.autocast("cuda", dtype=amp_dtype) if amp != "off" else nullcontext():
                        logits = model(*tensors)
                    batch = logits.float().sigmoid().cpu().numpy()
                    if batch.shape != (len(places), TILE_SIZE, TILE_SIZE):
                        raise ValueError(f"Unexpected detector output shape: {batch.shape}.")
                    for probs, (col, w, valid) in zip(batch, places, strict=True):
                        probs = probs[:h, :w]
                        if not np.isfinite(probs[valid]).all():
                            raise ValueError("Non-finite detector scores; try --amp off.")
                        win = np.s_[:h, col:col + w]
                        weight = kernel[:h, :w] * valid
                        sums[win] += np.where(valid, probs, 0.0) * weight
                        weights[win] += weight
                done += len(cols[start:start + batch_size])
                if done == total or done % 100 < batch_size:
                    print(f"  {done:,}/{total:,} tiles", flush=True)
            # Only emit rows that future tiles cannot touch.
            end = h if row == rows[-1] else stride
            covered = weights[:end] > 0
            scores = np.divide(sums[:end], weights[:end], out=np.zeros_like(sums[:end]), where=covered)
            changed = covered & (scores >= threshold)
            if output is not None:
                output[row:row + end, :, 0] = changed * 255
                output[row:row + end, :, 1] = covered * 255
            if score_png is not None:
                score_png[row:row + end] = np.rint(scores * 65535).astype(np.uint16)
            if alpha is not None:
                alpha[row:row + end] = score_alpha(scores)
            valid_count += int(covered.sum())
            changed_count += int(changed.sum())
            if overlap and row != rows[-1]:
                for strip in (sums, weights):
                    strip[:overlap] = strip[stride:].copy()
                    strip[overlap:] = 0
            else:
                sums.fill(0)
                weights.fill(0)
    return output, score_png, alpha, valid_count, changed_count


def new_png(path):
    if path.suffix.lower() != ".png":
        raise ValueError(f"Output needs a .png extension: {path}")
    if path.exists():
        raise FileExistsError(f"Choose a new output path: {path}")


def save_png(im, path, **kwargs):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents accidental overwrite.
    with path.open("xb") as stream:
        try:
            im.save(stream, format="PNG", **kwargs)
        except BaseException:
            stream.close()
            path.unlink()
            raise


def input_encodings(args, images, fallback, require_db=False):
    if (args.hh_db is None) != (args.hv_db is None) or (args.hh_db is not None and not args.assume_rg):
        raise ValueError("Supply both --hh-db and --hv-db together with --assume-rg.")
    supplied = ranges([args.hh_db, args.hv_db]) if args.hh_db is not None else None
    t0, t1 = images
    if t0.size != t1.size:
        raise ValueError(f"PNG sizes differ: {t0.size} vs {t1.size}. Align the scenes first.")
    if t0.width * t0.height > args.max_pixels:
        raise ValueError("Scene exceeds --max-pixels. Increase the limit if enough RAM is available.")
    if require_db and supplied is None and any(ENCODING_KEY not in im.info for im in images):
        raise ValueError("HH-delta rendering needs PNGs from 'pack' or their known linear "
                         "dB ranges via --assume-rg --hh-db LOW HIGH --hv-db LOW HIGH. "
                         "For browse PNGs and an existing mask, use 'render --max-delta-px 64'.")
    (e0, g0), (e1, g1) = (
        metadata(im, path, args.assume_rg, supplied, fallback)
        for im, path in zip(images, (args.t0, args.t1), strict=True)
    )
    if g0 is not None and g1 is not None and g0 != g1:
        raise ValueError("Packed PNGs refer to different source grids. Align them first.")
    return e0, e1


def render_paths(folder):
    return tuple(folder / name for name in (
        "t0_red_green.png", "t1_red_green.png",
        "t0_with_change.png", "t1_with_change.png",
    ))


def validate_outputs(paths):
    if len({p.resolve() for p in paths}) != len(paths):
        raise ValueError("Output paths must differ.")
    for path in paths:
        new_png(path)


def shared_display_ranges(images, encodings):
    width, height = images[0].size
    step = max(1, math.ceil(math.sqrt(width * height / 500_000)))
    result = np.array([np.min(encodings, axis=0)[:, 0], np.max(encodings, axis=0)[:, 1]]).T
    samples = []
    for im, encoding in zip(images, encodings, strict=True):
        for row in range(0, height, TILE_SIZE):
            db, valid = read_tile(im, (0, row, width, min(row + TILE_SIZE, height)), encoding)
            r = slice((-row) % step, None, step)
            samples.append(db[:, r, ::step][:, valid[r, ::step]])
    pooled = np.concatenate(samples, axis=1)
    if pooled.shape[1]:
        limits = np.percentile(pooled, (2, 98), axis=1).T
        usable = limits[:, 1] > limits[:, 0]
        result[usable] = limits[usable]
    return result


def render_scene(images, encodings, folder, *, alpha=None, selected=None, max_delta=10.0):
    width, height = images[0].size
    paths = render_paths(folder)
    low, high = shared_display_ranges(images, encodings).T[:, :, None, None]
    if alpha is None:
        alpha = np.zeros((height, width), np.uint8)
    blue = (0, 120, 255)
    # Build the layer in strips. In magnitude mode only detector-selected pixels get color.
    for row in range(0, height, TILE_SIZE):
        end = min(row + TILE_SIZE, height)
        box = (0, row, width, end)
        (x, v0), (y, v1) = (
            read_tile(im, box, enc) for im, enc in zip(images, encodings, strict=True)
        )
        if selected is not None:
            scores = np.where(v0 & v1 & selected[row:end], np.abs(y[0] - x[0]) / max_delta, 0)
            alpha[row:end] = score_alpha(scores)
        else:
            alpha[row:end][~(v0 & v1)] = 0
    layer = overlay_png(alpha, blue)
    for i, (im, enc) in enumerate(zip(images, encodings, strict=True)):
        scene = np.zeros((height, width, 4), np.uint8)
        for row in range(0, height, TILE_SIZE):
            end = min(row + TILE_SIZE, height)
            db, valid = read_tile(im, (0, row, width, end), enc)
            codes = np.rint(255 * np.clip((db - low) / (high - low), 0, 1)).astype(np.uint8)
            scene[row:end, :, :2] = codes.transpose(1, 2, 0)
            scene[row:end, :, 3] = valid * 255
        base = Image.fromarray(scene)
        save_png(base, paths[i])
        base.alpha_composite(layer.convert("RGBA"))
        save_png(base, paths[2 + i])
        del base, scene
    print(f"Saved four scene PNGs in {folder}.")


def infer(args):
    outputs = [p for p in (args.out, args.score_out, args.overlay_out) if p is not None]
    previews = render_paths(args.render_dir) if args.render_dir else ()
    if not (outputs or previews):
        raise ValueError("Specify --out, --overlay-out, --score-out, or --render-dir.")
    validate_outputs([*outputs, *previews])
    if not (0 < args.threshold < 1 and 0 <= args.overlap < TILE_SIZE and args.batch_size > 0):
        raise ValueError("Threshold must be in (0,1), overlap in [0,223], and batch size positive.")
    use_delta = args.render_dir is not None and args.render_mode == "hh-delta"
    payload, clip = checkpoint_spec(args.checkpoint)
    stats = payload["stats"]
    mean = np.asarray(stats["mean"], np.float32).reshape(2, 1, 1)
    std = np.asarray(stats["std"], np.float32).reshape(2, 1, 1)
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError("Checkpoint contains invalid normalization statistics.")
    device = torch.device(args.device if args.device != "auto" else "cuda" if torch.cuda.is_available() else "cpu")
    if device.type not in {"cpu", "cuda"} or (device.type == "cuda" and not torch.cuda.is_available()):
        raise ValueError("Choose an available CPU or CUDA device.")
    amp = args.amp
    if amp == "auto":
        amp = "off"
        if device.type == "cuda":
            with torch.cuda.device(device):
                amp = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    if amp != "off" and device.type != "cuda":
        raise ValueError("Use --amp off on CPU.")
    Image.MAX_IMAGE_PIXELS = args.max_pixels
    with Image.open(args.t0) as t0, Image.open(args.t1) as t1:
        images = (t0, t1)
        encodings = input_encodings(args, images, clip, require_db=use_delta)
        model = ChangeModel(TILE_SIZE, payload["arch"], payload["head_width"], grad_checkpoint=False)
        model.load_state_dict(payload["model"], strict=True)
        del payload
        model.to(device).eval()
        mask, scores, alpha, valid, changed = infer_scene(
            model, images, encodings, mean, std, clip, device, amp,
            args.batch_size, args.overlap, args.threshold, args.score_out is not None,
            save_overlay=args.overlay_out is not None or (bool(previews) and not use_delta),
            save_mask=args.out is not None or use_delta,
        )
        if previews:
            del model
            render_scene(images, encodings, args.render_dir,
                         alpha=None if use_delta else alpha,
                         selected=mask[..., 0] != 0 if use_delta else None, max_delta=args.max_delta_db)
    if args.out is not None:
        save_png(Image.fromarray(mask), args.out)
    if scores is not None:
        save_png(Image.fromarray(scores), args.score_out)
    if args.overlay_out is not None:
        save_png(overlay_png(alpha), args.overlay_out)
    if outputs:
        print("Saved " + ", ".join(map(str, outputs)))
    print(f"{changed:,} pixels at or above {args.threshold:g} / {valid:,} valid pixels.")
    if not valid:
        print("Warning: no valid pixels in common; output is transparent.")


def render(args):
    validate_outputs(render_paths(args.render_dir))
    Image.MAX_IMAGE_PIXELS = args.max_pixels
    browse_delta = args.max_delta_px is not None
    if browse_delta and (args.change_mask is None or args.hh_db is not None or args.hv_db is not None):
        raise ValueError("--max-delta-px requires --change-mask and cannot be combined with dB ranges.")
    with Image.open(args.t0) as t0, Image.open(args.t1) as t1:
        images = (t0, t1)
        if browse_delta and any(ENCODING_KEY in im.info for im in images):
            raise ValueError("--max-delta-px is for untagged browse PNGs. For packed PNGs, omit it to use dB.")
        encodings = input_encodings(args, images, ranges([[0, 255], [0, 255]]),
                                    require_db=args.change_mask is not None and not browse_delta)
        with Image.open(args.change_mask or args.change_score) as change:
            if change.format != "PNG" or change.size != t0.size:
                raise ValueError("The change PNG must have the same dimensions as T0 and T1.")
            if args.change_mask:
                if change.mode not in ("1", "L", "LA"):
                    raise ValueError("--change-mask expects the original binary PNG.")
                gray = np.asarray(change.convert("L"))
                if np.any((gray != 0) & (gray != 1) & (gray != 255)):
                    raise ValueError("--change-mask expects only zero and binary foreground values.")
                selected = gray > 0
                if change.mode == "LA":
                    selected &= np.asarray(change.getchannel("A")) != 0
                alpha = None
            else:
                if change.mode not in ("I", "I;16", "I;16B", "I;16L"):
                    raise ValueError("--change-score expects the 16-bit PNG from --score-out.")
                raw = np.asarray(change)
                if raw.min() < 0 or raw.max() > 65535:
                    raise ValueError("Expected 16-bit scores in [0, 65535].")
                alpha = np.empty(raw.shape, np.uint8)
                for row in range(0, change.height, TILE_SIZE):
                    part = raw[row:row + TILE_SIZE]
                    alpha[row:row + TILE_SIZE] = score_alpha(part.astype(np.float32) / 65535)
                selected = None
        if browse_delta:
            print(f"Browse view: full opacity at a red-channel difference of {args.max_delta_px:g} / 255.")
        render_scene(images, encodings, args.render_dir, alpha=alpha, selected=selected,
                     max_delta=args.max_delta_px if browse_delta else args.max_delta_db)


def colorize(args):
    new_png(args.out)
    Image.MAX_IMAGE_PIXELS = args.max_pixels
    with Image.open(args.input) as im:
        if im.format != "PNG" or im.width * im.height > args.max_pixels:
            raise ValueError("Expected a PNG within --max-pixels.")
        if im.mode in ("I", "I;16", "I;16B", "I;16L"):
            raw = np.asarray(im)
            if raw.min() < 0 or raw.max() > 65535:
                raise ValueError("Expected 16-bit scores in [0, 65535].")
            alpha = np.empty(raw.shape, np.uint8)
            for start in range(0, im.height, TILE_SIZE):
                rows = raw[start:start + TILE_SIZE]
                alpha[start:start + TILE_SIZE] = score_alpha(rows.astype(np.float32) / 65535)
        elif im.mode in ("LA", "L"):
            gray = np.asarray(im.getchannel("L"))
            if np.any((gray != 0) & (gray != 255)):
                raise ValueError("Expected a binary mask or the 16-bit PNG from --score-out.")
            alpha = gray.copy()
            if im.mode == "LA":
                alpha[np.asarray(im.getchannel("A")) == 0] = 0
            print("Binary mask: blue pixels will be fully opaque. Use the score PNG for graduated opacity.")
        else:
            raise ValueError("Use the original grayscale score or binary mask PNG.")
    save_png(overlay_png(alpha), args.out)
    print(f"Saved {args.out}")


def pack(args):
    try:
        import rasterio
    except ImportError as exc:
        raise ValueError("'pack' requires rasterio (your original NISAR environment uses it).") from exc
    new_png(args.out)
    payload, clip = checkpoint_spec(args.checkpoint)
    spacing = float(payload["stats"]["spacing_m"])
    del payload
    with rasterio.open(args.input) as src:
        if src.count != 2 or any(np.dtype(dt).kind != "f" for dt in src.dtypes):
            raise ValueError("Expected two floating-point dB bands, HH then HV.")
        for got, expected in zip(src.descriptions, (("HH", "HHHH"), ("HV", "HVHV")), strict=True):
            if got and got.upper() not in expected:
                raise ValueError(f"Unexpected band descriptions: {src.descriptions}.")
        tr = src.transform
        if (not src.crs or not src.crs.is_projected
                or not math.isclose(src.crs.linear_units_factor[1], 1)
                or tr.b != 0 or tr.d != 0 or tr.a <= 0 or tr.e >= 0
                or not all(math.isclose(s, spacing, rel_tol=1e-4) for s in (tr.a, -tr.e))):
            raise ValueError(f"Expected a north-up projected SAR raster with {spacing:g} m pixels.")
        if src.width * src.height > args.max_pixels:
            raise ValueError("Raster exceeds --max-pixels.")
        rgba = np.zeros((src.height, src.width, 4), dtype=np.uint8)
        for row in range(0, src.height, 224):
            h = min(224, src.height - row)
            db = src.read(window=rasterio.windows.Window(0, row, src.width, h),
                          masked=True, out_dtype="float32").filled(np.nan)
            valid = np.isfinite(db).all(axis=0)
            low, high = clip.T[:, :, None, None]
            db = np.where(valid[None], db, low)
            code = np.rint(255 * np.clip((db - low) / (high - low), 0, 1)).astype(np.uint8)
            rgba[row:row+h, :, :2] = code.transpose(1, 2, 0)
            rgba[row:row+h, :, 3] = valid * 255
        grid = {"crs": src.crs.to_string(), "transform": list(tr)[:6]}
    info = PngImagePlugin.PngInfo()
    info.add_text(ENCODING_KEY, json.dumps({"channel_order": ["HH", "HV"], "db_ranges": clip.tolist(), "grid": grid}))
    save_png(Image.fromarray(rgba), args.out, pnginfo=info)
    print(f"Saved {args.out} | {rgba.shape[1]}×{rgba.shape[0]}, red=HH, green=HV, alpha=valid.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("infer", help="Predict a change mask, scores, or a blue opacity overlay.")
    p.set_defaults(action=infer)
    p.add_argument("--threshold", type=float, default=.1)
    p.add_argument("--overlap", type=int, default=0, help="Set to 32 to blend predictions across tile borders.")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    p.add_argument("--amp", choices=["auto", "bf16", "fp16", "off"], default="auto")
    p.add_argument("--out", type=Path, help="Binary mask PNG.")
    p.add_argument("--score-out", type=Path, help="Optional 16-bit PNG of scores (0–65535).")
    p.add_argument("--overlay-out", type=Path, help="Blue overlay PNG with opacity equal to the model score.")
    p.add_argument("--render-mode", choices=("hh-delta", "score"), default="hh-delta",
                   help="What controls the blue opacity in --render-dir (default: hh-delta).")
    p.add_argument("--render-dir", type=Path, help="Save both scenes with and without the blue overlay.")
    r = commands.add_parser("render", help="Render before/after PNGs with an existing change map.")
    r.set_defaults(action=render)
    r.add_argument("--render-dir", type=Path, required=True)
    opacity = r.add_mutually_exclusive_group()
    opacity.add_argument("--max-delta-px", type=float,
                         help="Red-channel difference at full opacity for untagged browse PNGs (0–255).")
    for sub in (p, opacity):
        sub.add_argument("--max-delta-db", type=float, default=10.0, help="HH change in dB at full opacity.")
    source = r.add_mutually_exclusive_group(required=True)
    source.add_argument("--change-mask", type=Path, help="Binary mask for the QGIS-style HH difference.")
    source.add_argument("--change-score", type=Path, help="16-bit model scores for score-based opacity.")
    for sub in (p, r):
        sub.add_argument("--t0", type=Path, required=True)
        sub.add_argument("--t1", type=Path, required=True)
        sub.add_argument("--assume-rg", action="store_true", help="Allow untagged HH/HV PNGs.")
        sub.add_argument("--hh-db", type=float, nargs=2, metavar=("LOW", "HIGH"))
        sub.add_argument("--hv-db", type=float, nargs=2, metavar=("LOW", "HIGH"))
    q = commands.add_parser("pack", help="Pack a two-band HH/HV dB GeoTIFF into a compatible RGBA PNG.")
    q.set_defaults(action=pack)
    q.add_argument("--input", type=Path, required=True)
    q.add_argument("--out", type=Path, required=True)
    c = commands.add_parser("colorize", help="Color an existing score or binary mask PNG without inference.")
    c.set_defaults(action=colorize)
    c.add_argument("--input", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    for sub in (p, q):
        sub.add_argument("--checkpoint", type=Path, required=True)
    for sub in (p, q, c, r):
        sub.add_argument("--max-pixels", type=int, default=400_000_000)
    args = parser.parse_args()
    if args.max_pixels <= 0:
        parser.error("--max-pixels must be positive.")
    if args.command in ("infer", "render") and (not math.isfinite(args.max_delta_db) or args.max_delta_db <= 0):
        parser.error("--max-delta-db must be positive and finite.")
    if args.command == "render" and args.max_delta_px is not None and not 0 < args.max_delta_px <= 255:
        parser.error("--max-delta-px must be greater than 0 and no more than 255.")
    try:
        args.action(args)
    except (OSError, ValueError, KeyError, Image.DecompressionBombError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
