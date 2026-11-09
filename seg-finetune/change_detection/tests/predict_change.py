#!/usr/bin/env python3
"""Predict georeferenced tiles, including tiles with no pseudo labels; export stitch-compatible manifest."""
import argparse
import contextlib
import csv
from pathlib import Path
import numpy as np
import rasterio
import torch
from change_model import ChangeModel,load_payload
from finetune_change import read_rows
import nisar_label_pairs as prep


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--pair-id',nargs='+')
    p.add_argument('--split',choices=['train','val','test','all'],default='val')
    p.add_argument('--threshold',type=float,default=.5)
    p.add_argument('--amp',choices=['bf16','fp16','off'],default='bf16')
    a=p.parse_args()
    if not 0<a.threshold<1: p.error('Threshold must be between 0 and 1')
    if a.out.exists(): p.error('Use a new output directory')
    saved=load_payload(a.checkpoint)
    if saved.get('format')!='nisar_change_v1': p.error('Expected a change checkpoint')
    stats=saved['stats'];size=saved['size']
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type=='cpu' and a.amp!='off': p.error('Use --amp off on CPU')
    model=ChangeModel(size,saved['arch'],saved['head_width'])
    model.load_state_dict(saved['model'],strict=True);del saved
    model.to(device).eval()
    mean=np.asarray(stats['mean'],np.float32).reshape(2,1,1)
    std=np.asarray(stats['std'],np.float32).reshape(2,1,1)
    import json
    from finetune_change import check_stats
    check_stats(stats,json.loads((a.data/'stats.json').read_text()))
    rows=read_rows(a.data/'dataset_manifest.csv')
    if a.pair_id and set(a.pair_id)-{r['pair_id'] for r in rows}: p.error('Unknown pair ID')
    rows=[r for r in rows if (a.split=='all' or r['split']==a.split) and (not a.pair_id or r['pair_id'] in a.pair_id)]
    if not rows: p.error('No matching tiles; --split all allows any selected pair')
    a.out.mkdir(parents=True)
    amp=(lambda:contextlib.nullcontext()) if a.amp=='off' else (lambda:torch.autocast('cuda',dtype=torch.bfloat16 if a.amp=='bf16' else torch.float16))
    with (a.out/'prediction_manifest.csv').open('w',newline='') as stream:
        fields=list(rows[0])+['probability'];writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for i,row in enumerate(rows):
            arrays=[];grid=None
            for k in ('t0','t1','valid'):
                with rasterio.open(row[k]) as src:
                    g=(src.shape,src.transform,src.crs)
                    if grid is not None and g!=grid: raise ValueError('Input grid mismatch')
                    grid=g;arrays.append(src.read())
            x,y,v=arrays;valid=(v[0]==1)&np.isfinite(x).all(0)&np.isfinite(y).all(0)
            tensors=[torch.from_numpy(np.where(valid[None],(z-mean)/std,0).astype(np.float32))[None].to(device) for z in (x,y)]
            with torch.no_grad(),amp(): prob=model(*tensors).float().sigmoid()[0].cpu().numpy()
            prob[~valid]=np.nan;mask=np.where(valid,prob>=a.threshold,255).astype(np.uint8)
            dest=a.out/row['pair_id']/row['crop_id'];dest.mkdir(parents=True)
            prep.tif(dest/'probability.tif',prob,grid[1],grid[2].to_epsg(),np.nan)
            prep.tif(dest/'prediction.tif',mask,grid[1],grid[2].to_epsg(),255)
            row=dict(row,label=str((dest/'prediction.tif').resolve()),label_status='model_prediction',probability=str((dest/'probability.tif').resolve()))
            writer.writerow(row);stream.flush()
            if (i+1)%25==0: print(f'Predicted {i+1}/{len(rows)}',flush=True)
    print(f'Saved {len(rows)} tiles. Scores are not calibrated probabilities of real-world events.')

if __name__=='__main__': main()
