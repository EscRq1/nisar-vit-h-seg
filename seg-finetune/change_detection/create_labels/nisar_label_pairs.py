#!/usr/bin/env python3
"""Export compatible NISAR pair crops for review; rasterize reviewed change labels.
No learned model, automatic ground truth, or image-registration correction is applied.
"""
import argparse
import csv
import json
import re
from pathlib import Path

import h5py
import numpy as np
import rasterio
from affine import Affine
from rasterio.features import rasterize


def grid(h5, stats):
    g = h5[f"/science/LSAR/GCOV/grids/frequency{stats['frequency']}"]
    shape = g[stats['channels'][0]].shape
    for channel in stats['channels']:
        d = g[channel]
        if d.shape != shape or d.ndim != 2 or d.dtype.kind != 'f':
            raise ValueError(f'Expected matching real power arrays: {channel}')
    x = np.asarray(g['xCoordinates'][:], dtype=float)
    y = np.asarray(g['yCoordinates'][:], dtype=float)
    dx, dy = float(g['xCoordinateSpacing'][()]), float(g['yCoordinateSpacing'][()])
    if dx <= 0 or dy >= 0 or len(x) != shape[1] or len(y) != shape[0]:
        raise ValueError('Expected regular north-up grid with positive x / negative y spacing')
    if not np.allclose(np.diff(x), dx, atol=1e-5, rtol=0) or not np.allclose(np.diff(y), dy, atol=1e-5, rtol=0):
        raise ValueError('Irregular coordinates')
    epsg = int(g['projection'][()])
    if not (32601 <= epsg <= 32660 or 32701 <= epsg <= 32760 or epsg in (3413, 3031)):
        raise ValueError('Unsupported projection; verify metric units before resampling')
    transform = Affine(dx, 0, x[0]-dx/2, 0, dy, y[0]-dy/2)
    return g, shape, transform, epsg


def factors(transform, spacing):
    f = [spacing / abs(transform.e), spacing / transform.a]
    if any(v < 1 or not np.isclose(v, round(v), atol=1e-6, rtol=0) for v in f):
        raise ValueError('Cache spacing must be an integer multiple of native spacing')
    return tuple(round(v) for v in f)


def read_db(g, channels, row, col, size, fy, fx, clip):
    arrays, masks = [], []
    for channel in channels:
        ds = g[channel]
        a = np.asarray(ds[row:row+size*fy, col:col+size*fx], dtype=np.float32)
        good = np.isfinite(a) & (a > 0)
        fills = [ds.fillvalue]
        for key in ('_FillValue', 'missing_value'):
            if key in ds.attrs:
                fills.extend(np.asarray(ds.attrs[key]).reshape(-1).tolist())
        for v in fills:
            if v is not None and np.isfinite(v):
                good &= a != v
        arrays.append(a)
        masks.append(good)
    valid_native = np.logical_and.reduce(masks)
    power = np.where(valid_native[None], np.stack(arrays), 0)
    power = power.reshape(len(channels), size, fy, size, fx).mean((2, 4), dtype=np.float64)
    valid = valid_native.reshape(size, fy, size, fx).all((1, 3))
    valid &= np.isfinite(power).all(0) & (power > 0).all(0)
    db = np.clip(10*np.log10(np.where(valid[None], power, 1)), *clip).astype(np.float32)
    db[:, ~valid] = np.nan
    return db, valid


def tif(path, data, transform, epsg, nodata, descriptions=None):
    if data.ndim == 2:
        data = data[None]
    with rasterio.open(path, 'w', driver='GTiff', height=data.shape[1], width=data.shape[2],
                       count=len(data), dtype=data.dtype, crs=f'EPSG:{epsg}', transform=transform,
                       nodata=nodata, compress='deflate') as dst:
        dst.write(data)
        for i, name in enumerate(descriptions or [], 1):
            dst.set_band_description(i, name)


def preview(path, a, b, valid, channels):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    step = max(1, a.shape[1]//1000)
    a, b, valid = a[:, ::step, ::step], b[:, ::step, ::step], valid[::step, ::step]
    fig, axes = plt.subplots(len(channels), 3, figsize=(13, 8), squeeze=False)
    for c, name in enumerate(channels):
        values = np.concatenate([a[c][valid], b[c][valid]])
        lo, hi = np.percentile(values, [2, 98])
        if hi <= lo:
            hi = lo + 1
        for k, image in enumerate([a[c], b[c]]):
            axes[c,k].imshow(np.where(valid, image, np.nan), cmap='gray', vmin=lo, vmax=hi)
            axes[c,k].set_title(f'{name} — T{k} (shared dB scale)')
        m = axes[c,2].imshow(np.where(valid, b[c]-a[c], np.nan), cmap='RdBu_r', vmin=-5, vmax=5)
        axes[c,2].set_title(f'{name} — T1 minus T0; NOT a label')
        fig.colorbar(m, ax=axes[c,2], label='dB', shrink=0.75)
        for ax in axes[c]:
            ax.axis('off')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def prepare(args):
    stats_path = Path(args.stats)
    if stats_path.is_dir():
        stats_path /= 'stats.json'
    stats = json.loads(stats_path.read_text())
    if len(stats['channels']) != 2 or args.size < 1 or args.samples < 1 or not 0 < args.min_valid <= 1:
        raise ValueError('Need two channels, positive size/sample count, min-valid in (0,1]')
    with open(args.manifest, newline='') as f:
        rows = [r for r in csv.DictReader(f) if r['pair_id'] == args.pair_id]
    if len(rows) != 1:
        raise ValueError('pair-id must match exactly one manifest row')
    row = rows[0]
    if not re.fullmatch(r'[A-Za-z0-9_-]+', row['pair_id']):
        raise ValueError('Unsafe pair identifier')
    out = Path(args.out).resolve()
    if out.exists():
        raise ValueError('Choose a NEW output directory; existing annotations will not be overwritten')
    manifest_dir = Path(args.manifest).resolve().parent
    paths = [Path(row[k]) for k in ('t0_path', 't1_path')]
    paths = [p if p.is_absolute() else manifest_dir/p for p in paths]
    exports = []
    with h5py.File(paths[0], 'r') as h0, h5py.File(paths[1], 'r') as h1:
        g0, s0, tr0, crs0 = grid(h0, stats)
        g1, s1, tr1, crs1 = grid(h1, stats)
        if crs0 != crs1 or not np.allclose([tr0.a, tr0.e], [tr1.a, tr1.e], rtol=0, atol=1e-6):
            raise ValueError('Different CRS/native spacing. External linear-power resampling needed; no export performed.')
        fy, fx = factors(tr0, stats['spacing_m'])
        # Pixel offsets: position of the T0 origin within the T1 grid.
        dc = (tr0.c-tr1.c)/tr1.a
        dr = (tr0.f-tr1.f)/tr1.e
        if not np.allclose([dr, dc], np.round([dr, dc]), atol=1e-4, rtol=0):
            raise ValueError('Fractional-pixel grid offset. Resampling required; no export performed.')
        dr, dc = round(dr), round(dc)
        minr, minc = max(0, -dr), max(0, -dc)
        maxr = min(s0[0], s1[0]-dr) - args.size*fy
        maxc = min(s0[1], s1[1]-dc) - args.size*fx
        # Anchor averaging blocks at the native T0 grid origin.
        rlo, clo = int(np.ceil(minr/fy)), int(np.ceil(minc/fx))
        rhi, chi = maxr//fy, maxc//fx
        if rhi < rlo or chi < clo:
            raise ValueError('Insufficient common extent for this crop size')
        if (args.row is None) != (args.col is None):
            raise ValueError('Provide both --row and --col (native T0 pixel coordinates)')
        rng = np.random.default_rng(args.seed)
        seen = set()
        attempts = 1 if args.row is not None else args.samples*100
        for _ in range(attempts):
            r = args.row if args.row is not None else int(rng.integers(rlo, rhi+1))*fy
            c = args.col if args.col is not None else int(rng.integers(clo, chi+1))*fx
            if (r,c) in seen:
                continue
            seen.add((r,c))
            if r % fy or c % fx or not minr <= r <= maxr or not minc <= c <= maxc:
                raise ValueError('Requested crop outside overlap or not aligned to averaging blocks')
            a, v0 = read_db(g0, stats['channels'], r, c, args.size, fy, fx, stats['db_clip'])
            b, v1 = read_db(g1, stats['channels'], r+dr, c+dc, args.size, fy, fx, stats['db_clip'])
            valid = v0 & v1
            if valid.mean() < args.min_valid:
                continue
            crop = out/f'r{r}_c{c}'
            crop.mkdir(parents=True)
            a[:, ~valid], b[:, ~valid] = np.nan, np.nan
            transform = tr0 * Affine.translation(c,r) * Affine.scale(fx,fy)
            tif(crop/'t0_db.tif', a, transform, crs0, np.nan, stats['channels'])
            tif(crop/'t1_db.tif', b, transform, crs0, np.nan, stats['channels'])
            tif(crop/'difference_db.tif', b-a, transform, crs0, np.nan, stats['channels'])
            tif(crop/'valid.tif', valid.astype('uint8'), transform, crs0, 0)
            tif(crop/'label_template.tif', np.full(valid.shape,255,dtype='uint8'), transform, crs0, 255)
            preview(crop/'preview.png', a, b, valid, stats['channels'])
            exports.append(dict(pair_id=row['pair_id'], area=row['area'], split=row.get('split',''),
                crop_id=crop.name, t0=str(crop/'t0_db.tif'), t1=str(crop/'t1_db.tif'),
                valid=str(crop/'valid.tif'), label=str(crop/'label_template.tif'), label_status='unlabeled'))
            print(f'Exported {crop.name}: {valid.mean():.1%} jointly valid', flush=True)
            if len(exports) >= args.samples or args.row is not None:
                break
    if not exports:
        raise ValueError('No sufficiently valid crop found. Try a different seed, smaller --size, or explicit row/col.')
    with (out/'crop_manifest.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(exports[0])); w.writeheader(); w.writerows(exports)
    (out/'pair_info.json').write_text(json.dumps(dict(source_pair=row, preprocessing=stats,
        native_t1_offset=[dr,dc], seed=args.seed,
        note='Grid alignment only; inspect stable features. Crops may overlap: never split them randomly.'), indent=2))
    print(f'Saved {len(exports)} crops. ALL labels are 255; annotation is required.')


def labels(args):
    crop = Path(args.crop)
    output = crop/'labels_reviewed.tif'
    if output.exists():
        raise ValueError('labels_reviewed.tif exists; preserve it or rename it before exporting again')
    with rasterio.open(crop/'valid.tif') as src:
        valid = src.read(1).astype(bool)
        transform, crs = src.transform, src.crs
    # GeoJSON must use the raster CRS, explicitly acknowledged to avoid silent misplacement.
    if args.epsg != crs.to_epsg():
        raise ValueError('Polygon EPSG does not match crop CRS. Reproject vector in QGIS first.')
    fc = json.loads(Path(args.polygons).read_text())
    if fc.get('type') != 'FeatureCollection':
        raise ValueError('Expected a GeoJSON FeatureCollection')
    label = np.full(valid.shape,255,dtype='uint8')
    claimed = np.zeros(valid.shape, dtype=bool)
    for feature in fc['features']:
        value = feature.get('properties', {}).get('class')
        if value not in (0,1,255) or isinstance(value, bool):
            raise ValueError('Every polygon must have numeric class: 0, 1, or 255')
        geometry = feature.get('geometry')
        if not geometry or geometry.get('type') not in ('Polygon','MultiPolygon'):
            raise ValueError('Only Polygon and MultiPolygon features are supported')
        mask = rasterize([(geometry,1)], out_shape=valid.shape, transform=transform, dtype='uint8').astype(bool)
        if np.any(mask & claimed & (label != value)):
            raise ValueError('Conflicting polygons overlap. Make classes mutually exclusive in QGIS.')
        label[mask] = value
        claimed |= mask
    label[~valid] = 255
    if not np.any(label != 255):
        raise ValueError('No valid labeled pixels. Check polygon CRS and extent.')
    tif(output, label, transform, args.epsg, 255)
    counts = {str(v):int((label==v).sum()) for v in (0,1,255)}
    print(f'Saved {output}\nPixel counts: {counts}')
    print('Update crop_manifest.csv label path and label_status=reviewed AFTER reviewing the raster.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('prepare')
    a.add_argument('--manifest', required=True)
    a.add_argument('--pair-id', required=True)
    a.add_argument('--stats', required=True, help='Pretraining cache directory or stats.json')
    a.add_argument('--out', required=True)
    a.add_argument('--size', type=int, default=896, help='Output crop width/height, in coarse pixels')
    a.add_argument('--samples', type=int, default=8)
    a.add_argument('--min-valid', type=float, default=0.95)
    a.add_argument('--seed', type=int, default=42)
    a.add_argument('--row', type=int)
    a.add_argument('--col', type=int)
    a.set_defaults(func=prepare)
    a = sub.add_parser('labels')
    a.add_argument('--crop', required=True)
    a.add_argument('--polygons', required=True, help='Reviewed polygons with numeric class field')
    a.add_argument('--epsg', type=int, required=True, help='Explicit CRS of polygon coordinates')
    a.set_defaults(func=labels)
    args = p.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()
