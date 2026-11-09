#!/usr/bin/env python3
"""NISAR-only draft change labels. No checkpoint/GPU required."""
import argparse, csv, hashlib, json, sys, traceback
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import rasterio
from rasterio.features import rasterize
from scipy.ndimage import uniform_filter, binary_erosion, binary_dilation, label as components
import nisar_label_pairs as prep


def write_csv(path, rows):
    if not rows: return
    with Path(path).open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def read_crop(crop):
    arrays=[]; grid=None
    for name in ('t0_db.tif','t1_db.tif','valid.tif'):
        with rasterio.open(crop/name) as src:
            g=(src.shape,src.transform,src.crs)
            if grid is not None and g != grid: raise ValueError('Raster grids differ')
            grid=g; arrays.append(src.read())
    a,b,v=arrays
    if a.shape[0]!=2 or b.shape[0]!=2: raise ValueError('Expected two SAR channels')
    valid=(v[0]==1)&np.isfinite(a).all(0)&np.isfinite(b).all(0)
    return a,b,valid,grid


def draft(a,b,valid,clip,low,high,window,min_pixels):
    # Exclude clipped data: the true magnitude of a saturated change is unknown.
    usable=valid.copy()
    for arr in (a,b): usable &= ((arr>clip[0]+1e-4)&(arr<clip[1]-1e-4)).all(0)
    scores=[]; deltas=[]
    for w in (window,2*window-1):
        safe=binary_erosion(usable,structure=np.ones((w,w)),border_value=0)
        means=[]
        for arr in (a,b):
            power=np.where(usable[None],10.**(np.clip(arr,*clip)/10.),0.)
            sm=uniform_filter(power.astype(np.float64),size=(1,w,w),mode='constant')
            means.append(10*np.log10(np.maximum(sm,1e-30)))
        d=means[1]-means[0]; d[:,~safe]=np.nan
        deltas.append(d); scores.append(safe)
    safe=scores[0]&scores[1]
    ds=np.stack(deltas)
    # Require both channels and both scales to agree on substantial magnitude.
    # Each channel must keep its sign across scales; channels may have opposite signs.
    changed=safe & (np.abs(ds)>=high).all(axis=(0,1)) & (ds[0]*ds[1]>0).all(0)
    unchanged=safe & (np.abs(ds)<=low).all(axis=(0,1))
    cc,n=components(changed,structure=np.ones((3,3)))
    sizes=np.bincount(cc.ravel()); keep=sizes>=min_pixels;keep[0]=False
    changed=keep[cc]
    # Boundary and rejected components remain ignored, not relabeled unchanged.
    edge=binary_dilation(changed,iterations=1)&~binary_erosion(changed,iterations=1)
    lab=np.full(valid.shape,255,np.uint8)
    lab[unchanged&~edge]=0;lab[changed&~edge]=1
    score=np.max(np.abs(ds[0]),axis=0).astype(np.float32)
    score[~safe]=np.nan
    return lab,score,ds[0].astype(np.float32)


def preview(path,a,b,valid,lab,score):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    fig,axs=plt.subplots(2,3,figsize=(13,8))
    for ch in range(2):
        for date,arr in enumerate((a,b)):
            axs[ch,date].imshow(np.where(valid,arr[ch],np.nan),cmap='gray',vmin=-30,vmax=0)
            axs[ch,date].set_title(f'{["HHHH","HVHV"][ch]} t{date}, dB')
    im=axs[0,2].imshow(score,cmap='magma',vmin=0,vmax=8);axs[0,2].set_title('Maximum channel difference, dB');fig.colorbar(im,ax=axs[0,2],shrink=.7)
    display=np.where(lab==255,2,lab)
    axs[1,2].imshow(display,cmap=ListedColormap(['#438bb9','#ef5a50','#cccccc']),norm=BoundaryNorm([-.5,.5,1.5,2.5],3),interpolation='nearest')
    axs[1,2].set_title('Blue: unchanged; red: changed; grey: ignore')
    for ax in axs.flat:ax.axis('off')
    fig.suptitle('Draft SAR-change labels — review before use as evaluation truth');fig.tight_layout();fig.savefig(path,dpi=120);plt.close(fig)


def batch(args):
    out=Path(args.out).resolve()
    if out.exists():raise ValueError('Choose a NEW --out directory; existing labels will not be overwritten')
    if args.window<3 or args.window%2!=1 or not 0<=args.low_db<args.high_db or args.min_pixels<1:raise ValueError('Invalid thresholds/window/min-pixels')
    with open(args.manifest,newline='') as f: pairs=list(csv.DictReader(f))
    required={'pair_id','area','t0_path','t1_path'}
    if not pairs or not required.issubset(pairs[0]):raise ValueError('Invalid scene_pairs.csv schema')
    if len({r['pair_id'] for r in pairs})!=len(pairs):raise ValueError('Duplicate pair_id')
    if args.pair_id:
        unknown=set(args.pair_id)-{r['pair_id'] for r in pairs}
        if unknown:raise ValueError(f'Unknown pair IDs: {unknown}')
        pairs=[r for r in pairs if r['pair_id'] in args.pair_id]
    stats_path=Path(args.stats);stats_path=stats_path/'stats.json' if stats_path.is_dir() else stats_path
    stats=json.loads(stats_path.read_text())
    if stats['channels']!=['HHHH','HVHV']:raise ValueError('Expected HHHH, HVHV channel order')
    out.mkdir(parents=True)
    (out/'config.json').write_text(json.dumps({k:v for k,v in vars(args).items() if k != "func"},indent=2))
    manifest=[];errors=[];summaries=[]
    for i,row in enumerate(pairs):
        pid=row['pair_id'];print(f'[{i+1}/{len(pairs)}] {pid}',flush=True)
        try:
            import re
            if not re.fullmatch(r'[A-Za-z0-9_-]+',pid):raise ValueError('Unsafe pair_id')
            if args.prepared:
                found=[x.parent for x in Path(args.prepared).rglob('pair_info.json') if json.loads(x.read_text()).get('source_pair',{}).get('pair_id')==pid]
                if len(found)!=1:raise ValueError(f'Expected exactly one prepared directory for {pid}, found {len(found)}')
                source=found[0]
                old=json.loads((source/'pair_info.json').read_text())['preprocessing']
                if old!=stats:raise ValueError('Prepared crop statistics do not match --stats')
                crops=sorted(p.parent for p in source.glob('*/t0_db.tif'))
            else:
                source=out/'crops'/pid
                seed=(args.seed+int(hashlib.sha256(pid.encode()).hexdigest()[:8],16))%(2**32)
                prep.prepare(SimpleNamespace(stats=args.stats,manifest=args.manifest,pair_id=pid,out=str(source),size=args.size,samples=args.samples,min_valid=args.min_valid,seed=seed,row=None,col=None))
                crops=sorted(p.parent for p in source.glob('*/t0_db.tif'))
            if not crops:raise ValueError('No crops')
            for crop in crops:
                a,b,v,g=read_crop(crop)
                lab,score,delta=draft(a,b,v,stats['db_clip'],args.low_db,args.high_db,args.window,args.min_pixels)
                dest=out/'labels'/pid/crop.name;dest.mkdir(parents=True)
                prep.tif(dest/'labels_auto.tif',lab,g[1],g[2].to_epsg(),255)
                # Separate review copy; corrections never overwrite automatic labels.
                prep.tif(dest/'labels_editable.tif',lab,g[1],g[2].to_epsg(),255)
                prep.tif(dest/'change_score_db.tif',score,g[1],g[2].to_epsg(),np.nan)
                prep.tif(dest/'smoothed_difference_db.tif',delta,g[1],g[2].to_epsg(),np.nan,stats['channels'])
                preview(dest/'preview.png',a,b,v,lab,score)
                counts={str(k):int((lab==k).sum()) for k in (0,1,255)}
                info={'pair_id':pid,'crop_id':crop.name,'counts':counts,'label_status':'pseudo_unreviewed','note':'No class probabilities. Thresholds are heuristics, not calibrated confidence. A global brightness shift may label large areas.'}
                (dest/'report.json').write_text(json.dumps(info,indent=2))
                manifest.append(dict(pair_id=pid,area=row['area'],split=row.get('split',''),crop_id=crop.name,t0=str((crop/'t0_db.tif').resolve()),t1=str((crop/'t1_db.tif').resolve()),valid=str((crop/'valid.tif').resolve()),label=str(dest/'labels_editable.tif'),label_status='pseudo_unreviewed'))
                summaries.append(dict(pair_id=pid,crop_id=crop.name,unchanged=counts['0'],changed=counts['1'],ignored=counts['255'],review_priority='high' if counts['1']>0.5*lab.size or counts['0']+counts['1']==0 else 'normal'))
            if not args.prepared and len(crops)<args.samples:print(f'WARNING: only {len(crops)}/{args.samples} crops exported',flush=True)
        except Exception as e:
            errors.append({'pair_id':pid,'error':str(e)});traceback.print_exc()
        write_csv(out/'dataset_manifest.csv',manifest);write_csv(out/'review_summary.csv',summaries)
        (out/'errors.json').write_text(json.dumps(errors,indent=2))
    print(f'Done: {len(manifest)} crops; {len(errors)} pair errors. Output: {out}')
    if errors:raise SystemExit(1)


def correct(args):
    base=Path(args.label);dst=Path(args.out)
    if dst.exists():raise ValueError('Output exists; choose a new filename')
    with rasterio.open(base) as src:
        lab=src.read(1);tr=src.transform;crs=src.crs
    with rasterio.open(args.valid) as src:
        if src.shape!=lab.shape or src.transform!=tr or src.crs!=crs:raise ValueError('Validity grid mismatch')
        valid=src.read(1)==1
    if crs.to_epsg()!=args.epsg:raise ValueError('Explicit polygon EPSG does not match raster')
    if not np.isin(lab,[0,1,255]).all():raise ValueError('Invalid base label values')
    fc=json.loads(Path(args.polygons).read_text());claimed=np.zeros(lab.shape,bool)
    if fc.get('type')!='FeatureCollection':raise ValueError('Expected GeoJSON FeatureCollection')
    for feat in fc['features']:
        value=feat.get('properties',{}).get('class');geo=feat.get('geometry')
        if isinstance(value,bool) or value not in (0,1,255) or not geo or geo['type'] not in ('Polygon','MultiPolygon'):raise ValueError('Need polygons with numeric class 0,1,255')
        mask=rasterize([(geo,1)],out_shape=lab.shape,transform=tr,dtype='uint8').astype(bool)
        if np.any(mask&claimed&(lab!=value)):raise ValueError('Conflicting correction polygons overlap')
        lab[mask]=value;claimed|=mask
    if not np.any(claimed&valid):raise ValueError('No correction intersects valid pixels: check coordinates/CRS')
    lab[~valid]=255;dst.parent.mkdir(parents=True,exist_ok=True)
    prep.tif(dst,lab.astype('uint8'),tr,args.epsg,255)
    print(f'Saved {dst}. Unedited pixels retain draft labels; do not mark whole crop verified unless reviewed.')


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    b=sub.add_parser('batch');b.add_argument('--manifest',required=True);b.add_argument('--stats',required=True);b.add_argument('--out',required=True)
    b.add_argument('--prepared',help='Optional existing crop tree; pair_info.json must identify source pairs')
    b.add_argument('--pair-id',nargs='+');b.add_argument('--size',type=int,default=896);b.add_argument('--samples',type=int,default=8);b.add_argument('--min-valid',type=float,default=.95);b.add_argument('--seed',type=int,default=42)
    b.add_argument('--window',type=int,default=5);b.add_argument('--low-db',type=float,default=1.);b.add_argument('--high-db',type=float,default=3.);b.add_argument('--min-pixels',type=int,default=16);b.set_defaults(func=batch)
    c=sub.add_parser('correct');c.add_argument('--label',required=True);c.add_argument('--valid',required=True);c.add_argument('--polygons',required=True);c.add_argument('--epsg',type=int,required=True);c.add_argument('--out',required=True);c.set_defaults(func=correct)
    a=p.parse_args();a.func(a)
if __name__=='__main__':main()
