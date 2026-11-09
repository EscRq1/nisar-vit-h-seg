#!/usr/bin/env python3
"""Stitch georeferenced crop labels by pair. Uncovered/conflicting pixels stay 255."""
import argparse, csv, json, re
from pathlib import Path
from collections import defaultdict
import numpy as np
import rasterio
from affine import Affine
from rasterio.windows import Window


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--manifest',type=Path,required=True,help='dataset_manifest.csv; uses pair_id and label columns')
 p.add_argument('--out',type=Path,required=True,help='New output directory')
 p.add_argument('--pair-id',nargs='+',help='Optional pair IDs')
 p.add_argument('--reference',type=Path,help='Optional full-scene aligned reference raster; only with a single --pair-id')
 p.add_argument('--block-size',type=int,default=512)
 a=p.parse_args()
 if a.out.exists():p.error('Output directory exists; choose a new one')
 if a.block_size<1:p.error('--block-size must be positive')
 if a.reference and (not a.pair_id or len(a.pair_id)!=1):p.error('--reference requires exactly one --pair-id')
 with a.manifest.open(newline='') as f:
  reader=csv.DictReader(f)
  if not {'pair_id','label'}.issubset(reader.fieldnames or []):p.error('Manifest requires pair_id and label columns')
  rows=list(reader)
 groups=defaultdict(list);base=a.manifest.resolve().parent; images={}
 for row in rows:
  pid=row['pair_id'].strip()
  if a.pair_id and pid not in a.pair_id:continue
  if not re.fullmatch(r'[A-Za-z0-9_-]+',pid):p.error(f'Invalid pair ID: {pid!r}')
  path=Path(row['label']);path=path if path.is_absolute() else base/path
  if not path.is_file():p.error(f'Missing label: {path}')
  groups[pid].append((path.resolve(),row.get('label_status','unspecified')))
  images[path.resolve()]=[row.get('t0',''),row.get('t1','')]
 if a.pair_id and set(a.pair_id)-groups.keys():p.error('Some requested pair IDs are absent')
 if not groups:p.error('No crops found')
 a.out.mkdir(parents=True)
 summary=[]
 for pid,items in sorted(groups.items()):
  seen=set();spec=[];first=None
  for path,status in items:
   if path in seen:raise ValueError(f'Duplicate label path in manifest: {path}')
   seen.add(path)
   with rasterio.open(path) as ds:
    tr=ds.transform
    if ds.count!=1 or ds.crs is None or tr.b!=0 or tr.d!=0 or tr.a<=0 or tr.e>=0:raise ValueError(f'Expected single-band north-up georeferenced raster: {path}')
    if first is None:first=(ds.crs,tr)
    crs,anchor=first
    if ds.crs!=crs or not np.allclose([tr.a,tr.e],[anchor.a,anchor.e],atol=1e-8,rtol=0):raise ValueError(f'CRS or resolution mismatch: {path}')
    col=(tr.c-anchor.c)/anchor.a;row=(tr.f-anchor.f)/anchor.e
    if not np.allclose([row,col],np.round([row,col]),atol=1e-5,rtol=0):raise ValueError(f'Grid is not pixel-aligned: {path}')
    spec.append((path,int(round(row)),int(round(col)),ds.height,ds.width,status))
  if a.reference:
   with rasterio.open(a.reference) as ds:
    t=ds.transform; crs,anchor=first
    if ds.crs!=crs or t.b!=0 or t.d!=0 or not np.allclose([t.a,t.e],[anchor.a,anchor.e],atol=1e-8,rtol=0):raise ValueError('Reference CRS, orientation or resolution mismatch')
    r=(t.f-anchor.f)/anchor.e;c=(t.c-anchor.c)/anchor.a
    if not np.allclose([r,c],np.round([r,c]),atol=1e-5,rtol=0):raise ValueError('Reference pixel grid mismatch')
    r0,c0=int(round(r)),int(round(c));height,width=ds.height,ds.width
   if any(r<r0 or c<c0 or r+h>r0+height or c+w>c0+width for _,r,c,h,w,_ in spec):raise ValueError('One or more crops fall outside the reference raster')
  else:
   r0=min(x[1] for x in spec);c0=min(x[2] for x in spec)
   height=max(x[1]+x[3] for x in spec)-r0;width=max(x[2]+x[4] for x in spec)-c0
  transform=first[1]*Affine.translation(c0,r0)
  print(f'{pid}: {len(spec)} crops; output {width} x {height}',flush=True)
  dest=a.out/pid;dest.mkdir()
  prof=dict(driver='GTiff',height=height,width=width,count=1,crs=first[0],transform=transform,compress='deflate',BIGTIFF='IF_SAFER',tiled=True)
  counts={'unchanged':0,'changed':0,'ignored_or_uncovered':0,'conflicting':0,'covered_by_crop':0}
  with rasterio.open(dest/'labels_stitched.tif','w',**prof,dtype='uint8',nodata=255) as out, rasterio.open(dest/'coverage_count.tif','w',**prof,dtype='uint32') as cover, rasterio.open(dest/'conflicts.tif','w',**prof,dtype='uint8') as conflict:
   for y in range(0,height,a.block_size):
    for x in range(0,width,a.block_size):
     h=min(a.block_size,height-y);w=min(a.block_size,width-x)
     v0=np.zeros((h,w),bool);v1=np.zeros((h,w),bool);cov=np.zeros((h,w),np.uint32)
     for path,rr,cc,ch,cw,status in spec:
      ry=rr-r0;cx=cc-c0; ya=max(y,ry);yb=min(y+h,ry+ch);xa=max(x,cx);xb=min(x+w,cx+cw)
      if ya>=yb or xa>=xb:continue
      with rasterio.open(path) as ds:z=ds.read(1,window=Window(xa-cx,ya-ry,xb-xa,yb-ya))
      if not np.isin(z,[0,1,255]).all():raise ValueError(f'Unexpected label values in {path}; expected 0, 1, 255')
      sl=np.s_[ya-y:yb-y,xa-x:xb-x];cov[sl]+=1;v0[sl]|=z==0;v1[sl]|=z==1
     bad=v0&v1;result=np.full((h,w),255,np.uint8);result[v0&~v1]=0;result[v1&~v0]=1
     win=Window(x,y,w,h);out.write(result,1,window=win);cover.write(cov,1,window=win);conflict.write(bad.astype('uint8'),1,window=win)
     counts['unchanged']+=int((result==0).sum());counts['changed']+=int((result==1).sum());counts['ignored_or_uncovered']+=int((result==255).sum());counts['conflicting']+=int(bad.sum());counts['covered_by_crop']+=int((cov>0).sum())
  # The original input crops are mosaicked when the manifest includes both paths.
  for date_index,date in enumerate(('t0','t1')):
   if not all(images[path][date_index] for path, *_ in spec):
    print(f'{pid}: no complete {date} paths in manifest; skipping image mosaic',flush=True);continue
   sources=[];channels=None
   for path,rr,cc,ch,cw,status in spec:
    ip=Path(images[path][date_index]);ip=ip if ip.is_absolute() else base/ip
    with rasterio.open(ip) as src:
     expected=first[1]*Affine.translation(cc,rr)
     if src.crs!=first[0] or src.transform!=expected or (src.height,src.width)!=(ch,cw):raise ValueError(f'Image and label grid mismatch: {ip}')
     if channels is None:channels=src.count
     if src.count!=channels:raise ValueError(f'Image channel count mismatch: {ip}')
    sources.append((ip,rr-r0,cc-c0,ch,cw))
   imageprof=dict(prof);imageprof['count']=channels
   with rasterio.open(dest/f'{date}_stitched.tif','w',**imageprof,dtype='float32',nodata=np.nan) as dst:
    for y in range(0,height,a.block_size):
     for x in range(0,width,a.block_size):
      h=min(a.block_size,height-y);w=min(a.block_size,width-x);data=np.full((channels,h,w),np.nan,np.float32)
      for ip,ry,cx,ch,cw in sources:
       ya=max(y,ry);yb=min(y+h,ry+ch);xa=max(x,cx);xb=min(x+w,cx+cw)
       if ya>=yb or xa>=xb:continue
       with rasterio.open(ip) as src:z=src.read(window=Window(xa-cx,ya-ry,xb-xa,yb-ya),masked=True).astype('float32').filled(np.nan)
       sl=np.s_[:,ya-y:yb-y,xa-x:xb-x];old=data[sl];take=np.isfinite(z).all(0)&~np.isfinite(old).all(0);old[:,take]=z[:,take]
      dst.write(data,window=Window(x,y,w,h))
  info={'pair_id':pid,'crop_count':len(items),'width':width,'height':height,'crs':str(first[0]),'reference':str(a.reference.resolve()) if a.reference else None,'counts':counts,'label_status_counts':{s:sum(t==s for _,t in items) for _,s in items},'conflict_rule':'If any crop says 0 and another says 1, output 255; ignored crop pixels do not cast a vote.','note':'A mosaic of available labels, not an inference over the full scene. Uncovered pixels remain 255. Geometry is checked, not actual image registration. Image overlap uses first valid crop.'}
  (dest/'report.json').write_text(json.dumps(info,indent=2));summary.append(info)
 (a.out/'summary.json').write_text(json.dumps(summary,indent=2));print('Done.')
if __name__=='__main__':main()
