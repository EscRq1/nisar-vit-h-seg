#!/usr/bin/env python3
"""Compare two SAR MAE checkpoints with identical fixed validation masks."""
import argparse, contextlib, importlib.util, json, math
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch


def load_mae(path):
    spec=importlib.util.spec_from_file_location('user_nisar_mae',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--mae-script',type=Path,default=Path('nisar_mae_full.py'))
    p.add_argument('--num-tiles',type=int,default=64,help='Evenly spaced validation samples; use 0 for full validation')
    p.add_argument('--indices',type=int,nargs='+',help='Optional explicit validation indices, replacing --num-tiles')
    p.add_argument('--plots',type=int,default=4,help='Number of evaluated tiles to plot')
    p.add_argument('--seed',type=int,default=100042,help='Validation-mask seed; training seed 42 uses 100042')
    p.add_argument('--mask-ratio',type=float,default=.25)
    p.add_argument('--mask-repeats',type=int,default=1)
    p.add_argument('--amp',choices=['bf16','fp16','off'],default='bf16')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--expect-candidate-epoch',type=int,help='Abort if candidate is not this completed epoch')
    return p.parse_args()


def main():
    a=parse_args()
    if a.out.exists(): raise ValueError('Choose a new --out directory')
    if a.num_tiles<0 or a.plots<0 or a.mask_repeats<1 or not 0<a.mask_ratio<1: raise ValueError('Invalid test options')
    device=torch.device(a.device)
    if device.type=='cuda' and not torch.cuda.is_available(): raise ValueError('CUDA unavailable; use --device cpu --amp off')
    if device.type!='cuda' and a.amp!='off': raise ValueError('Use --amp off on CPU')
    if a.amp=='bf16' and not torch.cuda.is_bf16_supported(): raise ValueError('bf16 unsupported; use fp16 or off')
    amp=(lambda:contextlib.nullcontext()) if a.amp=='off' else lambda:torch.autocast('cuda',dtype=torch.bfloat16 if a.amp=='bf16' else torch.float16)
    mae=load_mae(a.mae_script.resolve())
    data=mae.Tiles(a.data,'val');n=len(data)
    if a.indices:
        indices=a.indices
        if len(indices)!=len(set(indices)) or min(indices)<0 or max(indices)>=n: raise ValueError('Indices must be unique and within validation split')
    elif a.num_tiles==0 or a.num_tiles>=n: indices=list(range(n))
    else: indices=np.linspace(0,n-1,a.num_tiles,dtype=int).tolist()
    display=[indices[i] for i in np.linspace(0,len(indices)-1,min(a.plots,len(indices)),dtype=int)] if a.plots else []
    ckpts=[]
    # Strict data provenance for candidate; baseline may come from original cache but must share transforms.
    for label,path in [('baseline',a.baseline),('candidate',a.candidate)]:
        state=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        if not all(k in state for k in ('model','stats','args','fingerprint')): raise ValueError(f'{label}: not a SAR MAE checkpoint')
        if not mae.preprocessing_equal(state['stats'],data.stats): raise ValueError(f'{label}: preprocessing differs from cache')
        if label=='candidate' and state['fingerprint']!=mae.cache_fingerprint(a.data): raise ValueError('Candidate was not trained with this exact cache')
        epoch=state.get('epoch');epoch=epoch+1 if isinstance(epoch,int) else None
        if label=='candidate' and a.expect_candidate_epoch is not None and epoch!=a.expect_candidate_epoch:
            raise ValueError(f'Candidate is completed epoch {epoch}, expected {a.expect_candidate_epoch}; last.pt may have advanced')
        ckpts.append(dict(label=label,path=str(path.resolve()),epoch_completed=epoch,step=state.get('step'),arch=state['args']['arch']))
        del state
    if ckpts[0]['arch']!=ckpts[1]['arch']: raise ValueError('Checkpoint architectures differ')
    a.out.mkdir(parents=True)
    results={};plot_data=defaultdict(dict)
    for info in ckpts:
        label=info['label'];print(f"Evaluating {label}, completed epoch {info['epoch_completed']}, {len(indices)} tiles x {a.mask_repeats} mask(s)",flush=True)
        state=torch.load(info['path'],map_location='cpu',weights_only=True,mmap=True)
        model=mae.SARMAE(data.stats['tile_size'],info['arch'],False)
        model.load_state_dict(state['model'],strict=True);del state
        model.to(device).eval();totals=np.zeros((2,2),np.float64);by_area=defaultdict(lambda:np.zeros((2,2),np.float64))
        size=data.stats['tile_size'];patch=model.patch_size;token_count=(size//patch)**2
        std=np.asarray(data.stats['std'],np.float64);mean=np.asarray(data.stats['mean'],np.float32)
        with torch.inference_mode():
          for j,idx in enumerate(indices):
            x,valid,_=data[idx];x=x[None].to(device);valid=valid[None].to(device)
            for rep in range(a.mask_repeats):
                # Repeat zero matches trainer seed convention when --seed 100042.
                seed=a.seed+idx+rep*1000000007
                noise=torch.rand(token_count,generator=torch.Generator().manual_seed(seed))[None].to(device)
                with amp(): _,_,_,pred,hidden=model(x,valid,a.mask_ratio,noise)
                predicted=model.unpatchify(pred.float())
                hidden_map=hidden.reshape(1,size//patch,size//patch).repeat_interleave(patch,1).repeat_interleave(patch,2).bool()
                score=(hidden_map&valid)[0].cpu().numpy()
                target=x[0].float().cpu().numpy();guess=predicted[0].cpu().numpy()
                area=data.records[idx].get('area','unknown')
                for c in range(2):
                    err=(guess[c][score].astype(np.float64)-target[c][score].astype(np.float64))
                    totals[c,0]+=np.dot(err,err);totals[c,1]+=len(err)
                    by_area[area][c,0]+=np.dot(err,err);by_area[area][c,1]+=len(err)
                if idx in display and rep==0:
                    plot_data[idx][label]=dict(target=target,guess=guess,valid=valid[0].cpu().numpy(),hidden=hidden_map[0].cpu().numpy())
            if (j+1)%50==0 or j+1==len(indices): print(f'  {label}: {j+1}/{len(indices)} tiles',flush=True)
        def metric(v):
            if (v[:,1]==0).any(): raise ValueError('No valid hidden pixels')
            channel=v[:,0]/v[:,1]
            return dict(normalized_mse=float(v[:,0].sum()/v[:,1].sum()),
                        db_rmse=float(np.sqrt((v[:,0]*std**2).sum()/v[:,1].sum())),
                        per_channel={data.stats['channels'][c]:dict(normalized_mse=float(channel[c]),db_rmse=float(np.sqrt(channel[c])*std[c])) for c in range(2)},
                        scored_channel_pixels=int(v[:,1].sum()))
        results[label]=dict(checkpoint=info,metrics=metric(totals),by_area={k:metric(v) for k,v in by_area.items()})
        del model
        if device.type=='cuda':torch.cuda.empty_cache()
    if display:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for idx in display:
            x=plot_data[idx]['baseline'];y=plot_data[idx]['candidate'];fig,axs=plt.subplots(2,5,figsize=(18,7),constrained_layout=True)
            for c in range(2):
                target=x['target'][c]*std[c]+mean[c]
                valid=x['valid'];hidden=x['hidden'];masked=valid&hidden
                base=x['guess'][c]*std[c]+mean[c];cand=y['guess'][c]*std[c]+mean[c]
                images=[np.where(valid,target,np.nan),np.where(valid&~hidden,target,np.nan),
                        np.where(valid,np.where(hidden,base,target),np.nan),np.where(valid,np.where(hidden,cand,target),np.nan),
                        np.where(masked,np.abs(cand-target)-np.abs(base-target),np.nan)]
                names=['Original','Visible input','Baseline','Candidate','Error change (candidate − baseline, dB)']
                limits=np.asarray(target[valid]);lo,hi=np.percentile(limits,[2,98]) if limits.size else (-30,0)
                for k,im in enumerate(images):
                    if k==4:
                        m=axs[c,k].imshow(im,cmap='RdBu_r',vmin=-5,vmax=5);fig.colorbar(m,ax=axs[c,k],fraction=.046)
                    else: axs[c,k].imshow(im,cmap='gray',vmin=lo,vmax=hi)
                    axs[c,k].set_title(f"{data.stats['channels'][c]}: {names[k]}");axs[c,k].axis('off')
            fig.suptitle(f'Validation tile {idx}, area {data.records[idx].get("area","unknown")}; same fixed mask, visible pixels copied')
            fig.savefig(a.out/f'val_{idx:06d}.png',dpi=130);plt.close(fig)
    base=results['baseline']['metrics']['normalized_mse'];cand=results['candidate']['metrics']['normalized_mse']
    report=dict(split='val',available_val_tiles=n,tiles=len(indices),indices=indices,mask_ratio=a.mask_ratio,mask_seed=a.seed,
                mask_repeats=a.mask_repeats,amp=a.amp,results=results,
                relative_mse_change_percent=100*(cand-base)/base if base else None,
                note='Pixel-weighted hidden-valid-pixel metrics; negative relative change is improvement. Evenly spaced tiles are not a geographically independent random sample. Validation may have been used for checkpoint selection; not independent test data.')
    (a.out/'comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report[k] for k in ('tiles','mask_ratio','results','relative_mse_change_percent')},indent=2),flush=True)
    print(f'Saved {a.out}/comparison.json and {len(display)} figures')

if __name__=='__main__': main()
