#!/usr/bin/env python3
"""Stream every tile in each pair's common GCOV grid; append without renumbering."""
import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
from pathlib import Path

import h5py
import numpy as np
import rasterio
from affine import Affine
from rasterio.transform import array_bounds
from rasterio.warp import transform_bounds
import nisar_label_pairs as prep
from nisar_auto_labels import draft

FIELDS = ['index','pair_id','area','split','crop_id','t0','t1','valid','label',
          'label_status','unchanged','changed','ignored']


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    os.replace(tmp, path)


def write_manifest(out, rows):
    target = out/'dataset_manifest.csv'
    tmp = target.with_suffix('.tmp')
    with tmp.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(rows)
    os.replace(tmp, target)


def read_window(g, shape, channels, row, col, height, width, fy, fx, clip):
    """Read a padded coarse window; average native linear power before taking dB."""
    nr, nc = height*fy, width*fx
    r0, c0 = max(0,row), max(0,col)
    r1, c1 = min(shape[0],row+nr), min(shape[1],col+nc)
    values = np.zeros((len(channels), nr, nc), np.float32)
    valid = np.zeros((nr,nc), bool)
    if r1 > r0 and c1 > c0:
        patches, masks = [], []
        for channel in channels:
            ds = g[channel]
            a = np.asarray(ds[r0:r1,c0:c1], dtype=np.float32)
            good = np.isfinite(a) & (a > 0)
            fills = [ds.fillvalue]
            for key in ('_FillValue','missing_value'):
                fills.extend(np.asarray(ds.attrs.get(key, [])).reshape(-1).tolist())
            for fill in fills:
                if fill is not None and np.isfinite(fill): good &= a != fill
            patches.append(a); masks.append(good)
        good = np.logical_and.reduce(masks)
        sl = (slice(r0-row,r1-row),slice(c0-col,c1-col))
        valid[sl] = good
        values[(slice(None),)+sl] = np.where(good[None],np.stack(patches),0)
    power = values.reshape(len(channels),height,fy,width,fx).mean((2,4), dtype=np.float64)
    valid = valid.reshape(height,fy,width,fx).all((1,3))
    valid &= np.isfinite(power).all(0) & (power > 0).all(0)
    db = np.clip(10*np.log10(np.where(valid[None],power,1)),*clip).astype(np.float32)
    db[:,~valid] = np.nan
    return db,valid


def inspect_pair(row, stats, size):
    with h5py.File(row['t0_path'],'r') as h0, h5py.File(row['t1_path'],'r') as h1:
        _,s0,tr0,epsg0 = prep.grid(h0,stats)
        _,s1,tr1,epsg1 = prep.grid(h1,stats)
    if epsg0 != epsg1 or not np.allclose([tr0.a,tr0.e],[tr1.a,tr1.e],atol=1e-6,rtol=0):
        raise ValueError(f"{row['pair_id']}: CRS/spacing mismatch; align linear-power data first")
    offset = np.array([(tr0.f-tr1.f)/tr1.e,(tr0.c-tr1.c)/tr1.a])
    if not np.allclose(offset,np.round(offset),atol=1e-4,rtol=0):
        raise ValueError(f"{row['pair_id']}: fractional pixel offset; registration/resampling required")
    dr,dc = map(int,np.round(offset))
    fy,fx = prep.factors(tr0,stats['spacing_m'])
    r0,c0 = math.ceil(max(0,-dr)/fy)*fy,math.ceil(max(0,-dc)/fx)*fx
    # Include the last partial averaging cell, but mark it invalid.
    h = math.ceil((min(s0[0],s1[0]-dr)-r0)/fy)
    w = math.ceil((min(s0[1],s1[1]-dc)-c0)/fx)
    if min(h,w) <= 0: raise ValueError('No common scene coverage')
    transform = tr0*Affine.translation(c0,r0)*Affine.scale(fx,fy)
    bounds = transform_bounds(f'EPSG:{epsg0}','EPSG:4326',*array_bounds(h,w,transform),densify_pts=41)
    if not np.isfinite(bounds).all() or bounds[2] < bounds[0]:
        raise ValueError('Dateline-crossing/invalid bounds need a dedicated geographic overlap check')
    return dict(height=h,width=w,rows=math.ceil(h/size),cols=math.ceil(w/size),
                native_row=r0,native_col=c0,dr=dr,dc=dc,fy=fy,fx=fx,
                transform=list(transform)[:6],epsg=epsg0,bounds=list(bounds))


def overlap(a,b):
    return min(a[2],b[2])>max(a[0],b[0]) and min(a[3],b[3])>max(a[1],b[1])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--stats',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--size',type=int,help='Defaults to checkpoint preprocessing tile_size')
    p.add_argument('--val-areas',nargs='+',default=[])
    p.add_argument('--pair-id',nargs='+')
    p.add_argument('--low-db',type=float,default=1.)
    p.add_argument('--high-db',type=float,default=3.)
    p.add_argument('--window',type=int,default=5)
    p.add_argument('--min-pixels',type=int,default=16)
    p.add_argument('--halo',type=int,default=32)
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    sp=a.stats/'stats.json' if a.stats.is_dir() else a.stats
    stats=json.loads(sp.read_text()); size=a.size or stats['tile_size']
    if stats['channels'] != ['HHHH','HVHV']: p.error('Expected HHHH, HVHV channel order')
    if size<14 or size%14: p.error('Size must be divisible by 14')
    if a.window<3 or a.window%2!=1 or not 0<=a.low_db<a.high_db or a.min_pixels<1 or a.halo<2*a.window:
        p.error('Invalid label settings; halo must be >= 2*window')
    with a.manifest.open(newline='') as f: pairs=list(csv.DictReader(f))
    if not pairs or not {'pair_id','area','t0_path','t1_path'}.issubset(pairs[0]): p.error('Invalid pairs CSV')
    if len({r['pair_id'] for r in pairs})!=len(pairs): p.error('Duplicate pair IDs')
    areas={r['area'] for r in pairs}
    if set(a.val_areas)-areas: p.error('Unknown validation area')
    if a.pair_id and set(a.pair_id)-{r['pair_id'] for r in pairs}: p.error('Unknown pair ID')
    # Assign whole areas before optional pair filtering.
    for r in pairs:
        if not re.fullmatch(r'[A-Za-z0-9_-]+',r['pair_id']): p.error('Unsafe pair ID')
        if a.val_areas:
            split='val' if r['area'] in a.val_areas else 'train'
            if r.get('split','').strip() and r['split'].strip()!=split: p.error('Explicit CSV split conflicts with area options')
            r['split']=split
        r['split']=r.get('split','').strip()
        if r['split'] not in ('train','val'): p.error('Set each CSV split to train or val, or supply --val-areas')
        for k in ('t0_path','t1_path'):
            path=Path(r[k]);path=path if path.is_absolute() else a.manifest.resolve().parent/path
            r[k]=str(path.resolve())
            if not path.is_file(): p.error(f'Missing HDF5: {path}')
    if a.pair_id: pairs=[r for r in pairs if r['pair_id'] in a.pair_id]
    out=a.out.resolve(); cfg=out/'build_config.json'; catalog_path=out/'pairs.json'
    config=dict(version=1,stats=stats,size=size,low_db=a.low_db,high_db=a.high_db,
                window=a.window,min_pixels=a.min_pixels,halo=a.halo,
                labeler_sha256=hashlib.sha256(Path(__file__).with_name('nisar_auto_labels.py').read_bytes()).hexdigest())
    if cfg.exists() and json.loads(cfg.read_text())!=config:
        p.error('Existing preprocessing/label configuration differs. Use a new output directory.')
    catalog=json.loads(catalog_path.read_text()) if catalog_path.exists() else {}
    if any(item['source']['split'] not in ('train','val') for item in catalog.values()):
        p.error('Existing dataset has test or invalid splits; use a new --out directory and assign its scenes to train or val')
    for row in pairs:
        pid=row['pair_id'];info=inspect_pair(row,stats,size)
        source_files={k:dict(path=row[k],size=Path(row[k]).stat().st_size,mtime_ns=Path(row[k]).stat().st_mtime_ns) for k in ('t0_path','t1_path')}
        item=dict(source=row,files=source_files,grid=info)
        if pid in catalog and catalog[pid]!=item: p.error(f'{pid}: source/grid/split changed; refusing unsafe resume')
        catalog[pid]=item
        print(f"{pid}: {info['width']} x {info['height']}; {info['rows']*info['cols']} possible tiles ({row['split']})",flush=True)
    items=list(catalog.items())
    for i,(pid,x) in enumerate(items):
        for qid,y in items[:i]:
            if x['source']['split'] != y['source']['split']:
                if x['source']['area']==y['source']['area'] or overlap(x['grid']['bounds'],y['grid']['bounds']):
                    p.error(f'Geographic split conflict: {pid} / {qid}. Keep this connected geography in one split.')
    possible=sum(x['grid']['rows']*x['grid']['cols'] for _,x in items)
    print(f'Total upper bound: {possible} tiles, ~{possible*size*size*19/1e9:.1f} GB uncompressed raster payload.',flush=True)
    print('Grid checks are not physical registration verification. Only fully invalid tiles are skipped.',flush=True)
    if a.dry_run: return
    out.mkdir(parents=True,exist_ok=True)
    # Prevent two writers from allocating duplicate stable indices.
    import fcntl
    with (out/'.build.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        atomic_json(cfg,config);atomic_json(out/'stats.json',stats);atomic_json(catalog_path,catalog)
        rows=[]
        if (out/'dataset_manifest.csv').exists():
            with (out/'dataset_manifest.csv').open(newline='') as f: rows=list(csv.DictReader(f))
        if any(r['split'] not in ('train','val') for r in rows):
            p.error('Existing dataset has test or invalid splits; use a new --out directory')
        known={(r['pair_id'],r['crop_id']):r for r in rows}
        used={int(r['index']) for r in rows}
        if len(known)!=len(rows) or len(used)!=len(rows): raise ValueError('Duplicate cached tile keys or indices')
        # Recover committed tile directories if extraction stopped before a CSV checkpoint.
        orphans=[]
        for record in (out/'crops').glob('*/*/record.json'):
            key=(record.parent.parent.name,record.parent.name)
            if key in known: continue
            rec=json.loads(record.read_text())
            for k in ('t0','t1','valid','label'):
                if not (out/rec[k]).is_file(): raise ValueError(f'Incomplete tile record: {record}')
            orphans.append(rec)
        for rec in sorted(orphans,key=lambda r:int(r['index'])):
            idx=int(rec['index'])
            if idx in used: raise ValueError('Conflicting recovered tile indices')
            rows.append(rec);used.add(idx);known[(rec['pair_id'],rec['crop_id'])]=rec
        if orphans: write_manifest(out,rows)
        next_index=max([int(r['index']) for r in rows],default=-1)+1
        for row in pairs:
            pid=row['pair_id'];info=catalog[pid]['grid'];tr=Affine(*info['transform'])
            pairdir=out/'crops'/pid;pairdir.mkdir(parents=True,exist_ok=True)
            skipped_path=pairdir/'skipped.json'
            skipped=set(json.loads(skipped_path.read_text())) if skipped_path.exists() else set()
            with h5py.File(row['t0_path'],'r') as h0,h5py.File(row['t1_path'],'r') as h1:
                g0,s0,_,_=prep.grid(h0,stats);g1,s1,_,_=prep.grid(h1,stats)
                for r in range(0,info['height'],size):
                    for c in range(0,info['width'],size):
                        cid=f'r{r:06d}_c{c:06d}';key=(pid,cid);dest=pairdir/cid
                        if key in known:
                            for k in ('t0','t1','valid','label'):
                                if not (out/known[key][k]).is_file(): raise ValueError(f'Missing cached file: {known[key][k]}')
                            continue
                        if cid in skipped: continue
                        if (dest/'record.json').exists():
                            # Recover a finished tile if interrupted just before CSV commit.
                            rec=json.loads((dest/'record.json').read_text())
                        else:
                            halo=a.halo;n=size+2*halo
                            nr=info['native_row']+(r-halo)*info['fy'];nc=info['native_col']+(c-halo)*info['fx']
                            x,v0=read_window(g0,s0,stats['channels'],nr,nc,n,n,info['fy'],info['fx'],stats['db_clip'])
                            y,v1=read_window(g1,s1,stats['channels'],nr+info['dr'],nc+info['dc'],n,n,info['fy'],info['fx'],stats['db_clip'])
                            v=v0&v1
                            rr=np.arange(n)+r-halo;cc=np.arange(n)+c-halo
                            v &= (rr[:,None]>=0)&(rr[:,None]<info['height'])&(cc[None]>=0)&(cc[None]<info['width'])
                            core=(slice(halo,halo+size),slice(halo,halo+size))
                            if not v[core].any():
                                skipped.add(cid);atomic_json(skipped_path,sorted(skipped));continue
                            lab,_,_=draft(x,y,v,stats['db_clip'],a.low_db,a.high_db,a.window,a.min_pixels)
                            lab=lab[core];v=v[core];x=x[(slice(None),)+core].copy();y=y[(slice(None),)+core].copy()
                            x[:,~v]=np.nan;y[:,~v]=np.nan
                            tmp=pairdir/(cid+'.partial')
                            if tmp.exists(): shutil.rmtree(tmp)
                            tmp.mkdir();tiletr=tr*Affine.translation(c,r)
                            prep.tif(tmp/'t0_db.tif',x,tiletr,info['epsg'],np.nan,stats['channels'])
                            prep.tif(tmp/'t1_db.tif',y,tiletr,info['epsg'],np.nan,stats['channels'])
                            prep.tif(tmp/'valid.tif',v.astype('uint8'),tiletr,info['epsg'],0)
                            for name in ('labels_auto.tif','labels_editable.tif'):
                                prep.tif(tmp/name,lab,tiletr,info['epsg'],255)
                                with rasterio.open(tmp/name,'r+') as ds:
                                    ds.write_colormap(1,{0:(0,0,0,255),1:(255,60,40,255),255:(0,0,0,0)})
                            rel=dest.relative_to(out)
                            rec=dict(index=next_index,pair_id=pid,area=row['area'],split=row['split'],crop_id=cid,
                                     t0=str(rel/'t0_db.tif'),t1=str(rel/'t1_db.tif'),valid=str(rel/'valid.tif'),
                                     label=str(rel/'labels_editable.tif'),label_status='pseudo_unreviewed',
                                     unchanged=int((lab==0).sum()),changed=int((lab==1).sum()),ignored=int((lab==255).sum()))
                            atomic_json(tmp/'record.json',rec)
                            if dest.exists(): raise ValueError(f'Incomplete existing tile: {dest}')
                            os.replace(tmp,dest)
                        rec['index']=next_index;next_index+=1
                        rows.append(rec);known[key]=rec
                        # A record.json commits each tile. Batch CSV rewrites avoid quadratic I/O per tile.
                        if len(rows)%250==0: write_manifest(out,rows)
                        if len(rows)%25==0: print(f'Saved {len(rows)} tiles; latest {pid}/{cid}',flush=True)
            atomic_json(skipped_path,sorted(skipped))
            write_manifest(out,rows)
        summary={s:sum(r['split']==s for r in rows) for s in ('train','val')}
        atomic_json(out/'build_summary.json',dict(tiles=len(rows),splits=summary,label_values={'0':'unchanged','1':'candidate SAR change','255':'ignore'}))
        print(f'Done: {summary}. Manifest: {out / "dataset_manifest.csv"}',flush=True)

if __name__=='__main__': main()
