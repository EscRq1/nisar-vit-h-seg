#!/usr/bin/env python3
"""Create a train/val reassignment of a completed NISAR tile cache without re-extraction.

Tile files are hard-linked, so source and output must share a filesystem and
linked tile contents must be treated as immutable. Metadata is independent.
"""
import argparse
import json
import math
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path

from pyproj import CRS, Geod, Transformer
from shapely.geometry import Polygon

GEOD = Geod(ellps='WGS84')


def footprint(scene):
    xmin,ymin,xmax,ymax = scene['bounds']
    corners=[(xmin,ymin),(xmax,ymin),(xmax,ymax),(xmin,ymax)]
    points=[]
    for i in range(4):
        ax,ay=corners[i]; bx,by=corners[(i+1)%4]
        points.extend((ax+(bx-ax)*k/8, ay+(by-ay)*k/8) for k in range(8))
    epsg=int(scene['epsg'])
    transform=Transformer.from_crs(CRS.from_epsg(epsg),4326,always_xy=True)
    lon,lat=transform.transform((xmin+xmax)/2,(ymin+ymax)/2)
    lons,lats=transform.transform([p[0] for p in points],[p[1] for p in points])
    if not all(math.isfinite(v) for v in [lon,lat,*lons,*lats]): raise ValueError('Invalid scene footprint coordinates')
    _,_,dist=GEOD.inv([lon]*len(lons),[lat]*len(lats),lons,lats)
    return (epsg,points,lon,lat,max(dist))


def overlaps(a,b,buffer_m):
    _,_,d=GEOD.inv(a[2],a[3],b[2],b[3])
    if d>a[4]+b[4]+buffer_m: return False
    local=CRS.from_proj4(f'+proj=aeqd +lat_0={a[3]} +lon_0={a[2]} +datum=WGS84 +units=m +no_defs')
    def polygon(s):
        trans=Transformer.from_crs(CRS.from_epsg(s[0]),local,always_xy=True)
        xs,ys=trans.transform([p[0] for p in s[1]],[p[1] for p in s[1]])
        result=Polygon(zip(xs,ys))
        if not result.is_valid: result=result.buffer(0)
        if result.is_empty: raise ValueError('Empty transformed footprint')
        return result
    return polygon(a).distance(polygon(b))<=buffer_m


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True,help='Completed cache; wait for preparation to finish')
    p.add_argument('--out',type=Path,required=True,help='New cache on the same filesystem')
    p.add_argument('--move-to-train',nargs='+',required=True,help='Whole existing validation area IDs')
    p.add_argument('--keep-val',nargs='+',required=True,help='All remaining validation area IDs')
    p.add_argument('--buffer-m',type=float,default=1000.)
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();src=a.data.resolve();out=a.out.resolve()
    if src==out or src in out.parents or out in src.parents: p.error('Input/output must be separate, non-nested directories')
    if out.exists(): p.error('Output already exists')
    if a.buffer_m<0: p.error('Buffer must be nonnegative')
    stats_path=src/'stats.json'
    if not stats_path.is_file(): p.error('No stats.json: cache preparation has not completed successfully. Wait for it to finish.')
    stats=json.loads(stats_path.read_text());scenes=json.loads((src/'scenes.json').read_text())
    if not stats.get('normalization_from_checkpoint_fingerprint'):
        print('WARNING: checkpoint normalization marker is absent; existing SAR checkpoint compatibility must be checked separately.',flush=True)
    moving=set(a.move_to_train);keeping=set(a.keep_val)
    if moving&keeping: p.error('An area cannot be in both lists')
    old_val={s['area'] for s in scenes if s['split']=='val'}
    if moving|keeping!=old_val: p.error(f'Move and keep lists must exactly partition current validation areas: {sorted(old_val)}')
    before=Counter((s['area'],s['split']) for s in scenes)
    for s in scenes:
        if s['area'] in moving:
            if s['split']!='val': p.error('Requested area contains a non-validation scene')
            s['split']='train'
    # Same geographic area must never be divided even if metadata are inconsistent.
    groups={}
    for s in scenes:
        if s['area'] in groups and groups[s['area']]!=s['split']: p.error('Area appears in both splits')
        groups[s['area']]=s['split']
    train=[s for s in scenes if s['split']=='train'];val=[s for s in scenes if s['split']=='val']
    if not train or not val: p.error('Both train and validation scenes are required')
    shapes={i:footprint(s) for i,s in enumerate(scenes)}
    for i,s in enumerate(scenes):
        for j in range(i):
            if scenes[j]['split']!=s['split'] and overlaps(shapes[i],shapes[j],a.buffer_m):
                p.error(f"Cross-split footprints overlap or are within buffer: {s['path']} and {scenes[j]['path']}")
    print('Scene assignment before change:',dict((f'{x}/{y}',n) for (x,y),n in sorted(before.items())),flush=True)
    print(f'After change: {len(train)} train scenes, {len(val)} validation scenes; buffer check passed.',flush=True)
    print('Footprint metadata checks are conservative and do not replace downstream holdout review.',flush=True)
    if a.dry_run: return
    out.parent.mkdir(parents=True,exist_ok=True)
    if os.stat(src).st_dev!=os.stat(out.parent).st_dev: p.error('Hard links require output on the same filesystem as input')
    stage=Path(tempfile.mkdtemp(prefix=out.name+'.partial-',dir=out.parent))
    try:
        seen=set();counts=Counter();moved=0
        with (src/'index.jsonl').open() as inp,(stage/'index.jsonl').open('x') as dst:
            for line in inp:
                if not line.strip(): continue
                row=json.loads(line);sid=row['scene_id']
                if not isinstance(sid,int) or not 0<=sid<len(scenes): raise ValueError('Invalid scene_id in index')
                scene=scenes[sid]
                if row['area']!=scene['area']: raise ValueError('Index area differs from scene metadata')
                if row['area'] in moving:
                    if row['split']!='val': raise ValueError('Moving index tile is not validation')
                    row['split']='train';moved+=1
                elif row['split']!=scene['split']: raise ValueError('Index split differs from scene metadata')
                rel=Path(row['file'])
                if rel.is_absolute() or '..' in rel.parts: raise ValueError('Unsafe tile path')
                tile=(src/rel).resolve()
                if src not in tile.parents or not tile.is_file(): raise ValueError(f'Missing or invalid tile: {tile}')
                if str(rel) in seen: raise ValueError(f'Duplicate tile file: {rel}')
                seen.add(str(rel))
                target=stage/rel;target.parent.mkdir(parents=True,exist_ok=True)
                os.link(tile,target)
                dst.write(json.dumps(row)+'\n');counts[row['split']]+=1
                if sum(counts.values())%50000==0: print(f'Linked {sum(counts.values()):,} tiles',flush=True)
        if not counts['train'] or not counts['val'] or not moved: raise ValueError('Result is missing tiles or nothing moved')
        # Do not recompute mean/std; checkpoint-normalization provenance remains intact.
        stats['split_tiles']=dict(counts)
        stats['resplit_note']='Mean/std and any train pixel/clipping statistics are inherited from before split reassignment; split_tiles is updated.'
        (stage/'scenes.json').write_text(json.dumps(scenes,indent=2)+'\n')
        (stage/'stats.json').write_text(json.dumps(stats,indent=2,allow_nan=False)+'\n')
        report=dict(moved_areas=sorted(moving),remaining_val_areas=sorted(keeping),moved_tiles=moved,
                    split_tiles=dict(counts),train_scenes=len(train),val_scenes=len(val),buffer_m=a.buffer_m,
                    normalization='unchanged',storage='hard links; tile contents shared with source',
                    note='Split-relative tile indices change. Old checkpoint validation loss is not directly comparable.')
        (stage/'resplit_report.json').write_text(json.dumps(report,indent=2)+'\n')
        # Original append lineage is intentionally not copied: index split metadata changed.
        if out.exists(): raise ValueError('Output appeared while processing; choose a different output')
        os.rename(stage,out)
        print(json.dumps(report,indent=2),flush=True)
    finally:
        if stage.exists(): shutil.rmtree(stage)

if __name__=='__main__': main()
