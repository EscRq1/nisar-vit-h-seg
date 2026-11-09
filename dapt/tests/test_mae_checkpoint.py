#!/usr/bin/env python3
"""Evaluate ONE SAR MAE checkpoint on masked cache tiles, without model comparison."""
import argparse
import contextlib
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


def load_script(path):
    spec=importlib.util.spec_from_file_location('nisar_pretraining_script',path)
    if not spec or not spec.loader: raise ValueError(f'Cannot import {path}')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    for name in ('Tiles','SARMAE','preprocessing_equal','cache_fingerprint'):
        if not hasattr(mod,name): raise ValueError(f'MAE script is missing {name}')
    return mod


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True,help='Prepared tile cache')
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True,help='New output directory')
    p.add_argument('--mae-script',type=Path,default=Path('nisar_mae_full.py'))
    p.add_argument('--split',choices=['train','val'],default='val')
    p.add_argument('--num-tiles',type=int,default=64,help='Evenly spaced sample; 0 means all tiles')
    p.add_argument('--indices',type=int,nargs='+',help='Specific zero-based indices within the chosen split')
    p.add_argument('--plots',type=int,default=4,help='Number of sampled tiles to visualize')
    p.add_argument('--seed',type=int,default=100042,help='Fixed mask seed; training seed 42 uses 100042 for validation')
    p.add_argument('--mask-ratio',type=float,default=.25)
    p.add_argument('--mask-repeats',type=int,default=1)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--amp',choices=['bf16','fp16','off'],default='bf16')
    p.add_argument('--allow-other-cache',action='store_true',help='Permit compatible normalization on another cache; geography must be checked separately')
    p.add_argument('--expect-epoch',type=int,help='Require this completed epoch number')
    a=p.parse_args()
    if a.out.exists(): p.error('Output exists; choose a new --out directory')
    if a.num_tiles<0 or a.plots<0 or a.mask_repeats<1 or not 0<a.mask_ratio<1: p.error('Invalid test settings')
    if a.seed<0: p.error('Seed must be nonnegative')
    device=torch.device(a.device)
    if device.type=='cuda' and not torch.cuda.is_available(): p.error('CUDA unavailable; use --device cpu --amp off')
    if device.type!='cuda' and a.amp!='off': p.error('Use --amp off on CPU')
    if device.type=='cuda':
        torch.cuda.set_device(device)
        if a.amp=='bf16' and not torch.cuda.is_bf16_supported(): p.error('BF16 unsupported; use --amp fp16 or off')
    amp=(lambda:contextlib.nullcontext()) if a.amp=='off' else lambda:torch.autocast('cuda',dtype=torch.bfloat16 if a.amp=='bf16' else torch.float16)
    mae=load_script(a.mae_script.resolve());data=mae.Tiles(a.data,a.split);n=len(data)
    if a.indices is not None:
        idx=a.indices
        if len(idx)!=len(set(idx)) or min(idx)<0 or max(idx)>=n: p.error('Indices must be unique and within the selected split')
    elif a.num_tiles==0 or a.num_tiles>=n: idx=list(range(n))
    else: idx=np.linspace(0,n-1,a.num_tiles,dtype=int).tolist()
    show={idx[i] for i in np.linspace(0,len(idx)-1,min(a.plots,len(idx)),dtype=int)} if a.plots else set()
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=True,mmap=True)
    if not all(k in state for k in ('model','stats','args','fingerprint')): raise ValueError('Expected a trained SAR MAE checkpoint')
    if not mae.preprocessing_equal(state['stats'],data.stats): raise ValueError('Checkpoint preprocessing/normalization does not match this cache')
    same_cache=state['fingerprint']==mae.cache_fingerprint(a.data)
    if not same_cache and not a.allow_other_cache: raise ValueError('Checkpoint and cache fingerprints differ. Use --allow-other-cache only after verifying preprocessing and geography.')
    saved_epoch=state.get('epoch');epoch=saved_epoch+1 if isinstance(saved_epoch,int) else None
    if a.expect_epoch is not None and epoch!=a.expect_epoch: raise ValueError(f'Checkpoint is completed epoch {epoch}, expected {a.expect_epoch}. last.pt may have advanced.')
    arch=state['args']['arch'];size=data.stats['tile_size']
    model=mae.SARMAE(size,arch,False);model.load_state_dict(state['model'],strict=True)
    step=state.get('step');del state
    model.to(device).eval()
    std=np.asarray(data.stats['std'],np.float64);mean=np.asarray(data.stats['mean'],np.float64)
    patch=model.patch_size;tokens=(size//patch)**2
    totals={'model':np.zeros((2,2),np.float64),'visible_mean':np.zeros((2,2),np.float64)}
    areas=defaultdict(lambda:np.zeros((2,2),np.float64));figures={}
    print(f'Testing {len(idx)} of {n} {a.split} tiles, {a.mask_repeats} mask(s), checkpoint completed epoch {epoch}',flush=True)
    with torch.inference_mode():
        for k,i in enumerate(idx):
            x,valid,_=data[i];x=x[None].to(device);valid=valid[None].to(device)
            for repeat in range(a.mask_repeats):
                noise=torch.rand(tokens,generator=torch.Generator().manual_seed(a.seed+i+repeat*1000000007))[None].to(device)
                with amp(): _,_,_,pred,hidden=model(x,valid,a.mask_ratio,noise)
                prediction=model.unpatchify(pred.float())[0].cpu().numpy();target=x[0].float().cpu().numpy()
                mask=hidden.reshape(1,size//patch,size//patch).repeat_interleave(patch,1).repeat_interleave(patch,2)[0].bool().cpu().numpy()
                valid_np=valid[0].cpu().numpy();scored=mask&valid_np;visible=(~mask)&valid_np
                if not visible.any(): raise ValueError(f'Tile {i} has no visible valid pixels')
                for c in range(2):
                    target_values=target[c][scored].astype(np.float64)
                    errors=prediction[c][scored].astype(np.float64)-target_values
                    mean_errors=float(target[c][visible].mean())-target_values
                    count=len(errors)
                    totals['model'][c]+=(np.dot(errors,errors),count)
                    totals['visible_mean'][c]+=(np.dot(mean_errors,mean_errors),count)
                    areas[data.records[i].get('area','unknown')][c]+=(np.dot(errors,errors),count)
                if i in show and repeat==0: figures[i]=(target,prediction,valid_np,mask)
            if (k+1)%50==0 or k+1==len(idx): print(f'Processed {k+1}/{len(idx)} tiles',flush=True)
    def metrics(v):
        if (v[:,1]<=0).any(): raise ValueError('No valid hidden pixels')
        mse=v[:,0]/v[:,1]
        return dict(normalized_mse=float(v[:,0].sum()/v[:,1].sum()),
            db_rmse=float(np.sqrt((v[:,0]*std**2).sum()/v[:,1].sum())),
            per_channel={data.stats['channels'][c]:dict(normalized_mse=float(mse[c]),db_rmse=float(np.sqrt(mse[c])*std[c])) for c in range(2)},
            scored_channel_pixels=int(v[:,1].sum()))
    result={k:metrics(v) for k,v in totals.items()}
    if result['visible_mean']['normalized_mse']>0:
        result['model']['error_reduction_vs_visible_mean_percent']=100*(1-result['model']['normalized_mse']/result['visible_mean']['normalized_mse'])
    report=dict(checkpoint=str(a.checkpoint.resolve()),checkpoint_epoch_completed=epoch,checkpoint_step=step,
        split=a.split,available_tiles=n,tiles=len(idx),indices=idx,mask_ratio=a.mask_ratio,mask_seed=a.seed,
        mask_repeats=a.mask_repeats,amp=a.amp,same_cache_fingerprint=same_cache,metrics=result,
        model_by_area={k:metrics(v) for k,v in areas.items()},
        note='Errors scored only on valid hidden pixels. Visible-mean is a simple non-model baseline. Validation may have been used for model selection; train tiles are not independent. Evenly spaced tiles are not a random geographic sample.')
    a.out.mkdir(parents=True)
    if figures:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for i,(target,prediction,valid,mask) in figures.items():
            fig,axs=plt.subplots(2,4,figsize=(16,8),constrained_layout=True)
            for c in range(2):
                truth=target[c]*std[c]+mean[c];guess=prediction[c]*std[c]+mean[c]
                hidden=mask&valid
                panels=[np.where(valid,truth,np.nan),np.where(valid&~mask,truth,np.nan),
                        np.where(valid,np.where(mask,guess,truth),np.nan),np.where(hidden,np.abs(guess-truth),np.nan)]
                valid_values=truth[valid];lo,hi=np.percentile(valid_values,[2,98]) if valid_values.size else (-30,0)
                if hi<=lo:hi=lo+1
                for j,arr in enumerate(panels):
                    if j==3:
                        img=axs[c,j].imshow(arr,cmap='magma',vmin=0,vmax=10);fig.colorbar(img,ax=axs[c,j],fraction=.046,label='absolute error (dB)')
                    else: axs[c,j].imshow(arr,cmap='gray',vmin=lo,vmax=hi)
                    axs[c,j].set_title(f"{data.stats['channels'][c]}: {['Original','Visible input','Reconstruction','Hidden-pixel error'][j]}");axs[c,j].axis('off')
            fig.suptitle(f'{a.split.upper()} tile {i}, area {data.records[i].get("area","unknown")}; visible pixels copied')
            fig.savefig(a.out/f'{a.split}_{i:06d}.png',dpi=130);plt.close(fig)
    (a.out/'metrics.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'checkpoint_epoch_completed':epoch,'tiles':len(idx),'metrics':result},indent=2),flush=True)
    print(f'Saved {a.out / "metrics.json"} and {len(figures)} image(s)',flush=True)

if __name__=='__main__': main()
