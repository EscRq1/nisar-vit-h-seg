#!/usr/bin/env python3
"""Fine-tune a SARMAE encoder for dense binary change segmentation (single GPU or DDP)."""
import argparse
import contextlib
import csv
import hashlib
import itertools
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset,DataLoader,DistributedSampler
from change_model import ChangeModel,load_mae_encoder,load_payload


def atomic_save(obj,path):
    tmp=path.with_suffix(path.suffix+'.tmp');torch.save(obj,tmp);os.replace(tmp,path)


def read_rows(path):
    with path.open(newline='') as f: rows=list(csv.DictReader(f))
    if not rows or not {'pair_id','area','split','crop_id','t0','t1','valid','label','label_status'}.issubset(rows[0]):
        raise ValueError('Invalid dataset_manifest.csv')
    seen=set();areas={};pairs={};paths={}
    for row in rows:
        key=(row['pair_id'],row['crop_id'])
        if key in seen: raise ValueError(f'Duplicate tile: {key}')
        seen.add(key)
        if row['split'] not in ('train','val','test'): raise ValueError('Every tile needs train/val/test split')
        for table,key2 in ((areas,row['area']),(pairs,row['pair_id'])):
            if key2 in table and table[key2]!=row['split']: raise ValueError(f'Cross-split group: {key2}')
            table[key2]=row['split']
        for k in ('t0','t1','valid','label'):
            p=Path(row[k]);p=p if p.is_absolute() else path.resolve().parent/p
            row[k]=str(p.resolve())
            if not p.is_file(): raise FileNotFoundError(p)
        for k in ('t0','t1'):
            if row[k] in paths and paths[row[k]]!=row['split']: raise ValueError('Same imagery in different splits')
            paths[row[k]]=row['split']
    return rows


def check_stats(saved,current):
    for k in ('frequency','channels','spacing_m','db_clip','mean','std'):
        if k not in saved or saved[k]!=current[k]: raise ValueError(f'Checkpoint preprocessing mismatch: {k}')


def data_fingerprint(path,rows):
    h=hashlib.sha256(path.read_bytes())
    for r in rows:
        for k in ('t0','t1','valid','label'):
            p=Path(r[k]);s=p.stat();h.update(f'{p}|{s.st_size}|{s.st_mtime_ns}'.encode())
    return h.hexdigest()


class Tiles(Dataset):
    def __init__(self,rows,stats,size,augment=False,seed=42):
        self.rows=[];self.stats=stats;self.size=size;self.augment=augment;self.seed=seed;self.epoch=0
        self.mean=np.asarray(stats['mean'],np.float32).reshape(2,1,1)
        self.std=np.asarray(stats['std'],np.float32).reshape(2,1,1)
        if not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or (self.std<=0).any(): raise ValueError('Invalid normalization')
        self.counts=np.zeros(3,np.int64)
        for row in rows:
            with rasterio.open(row['label']) as lab,rasterio.open(row['valid']) as valid:
                if lab.shape!=(size,size) or valid.shape!=lab.shape or valid.crs!=lab.crs or valid.transform!=lab.transform:
                    raise ValueError(f"Label grid mismatch: {row['label']}")
                y=lab.read(1);v=valid.read(1)==1
                if not np.isin(y,[0,1,255]).all(): raise ValueError('Label must contain only 0,1,255')
                y=np.where(v,y,255)
                counts=np.array([(y==k).sum() for k in (0,1,255)])
                self.counts+=counts
                if counts[:2].sum(): self.rows.append(row)
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        row=self.rows[i];arrays=[];grid=None
        for k in ('t0','t1','valid','label'):
            with rasterio.open(row[k]) as src:
                g=(src.shape,src.transform,src.crs)
                if grid is not None and g!=grid: raise ValueError(f'Input raster grid mismatch: {row[k]}')
                grid=g;arrays.append(src.read())
        a,b,v,y=arrays;y=y[0].astype(np.int64)
        if a.shape!=(2,self.size,self.size) or b.shape!=a.shape: raise ValueError('Input dimensions differ from checkpoint')
        good=(v[0]==1)&np.isfinite(a).all(0)&np.isfinite(b).all(0)
        y[~good]=255
        a=np.where(good[None],(a-self.mean)/self.std,0).astype(np.float32)
        b=np.where(good[None],(b-self.mean)/self.std,0).astype(np.float32)
        if self.augment:
            rng=np.random.default_rng(self.seed+self.epoch*1000003+i)
            rot=int(rng.integers(4));flip=bool(rng.integers(2))
            a,b,y=[np.rot90(x,rot,axes=(-2,-1)) for x in (a,b,y)]
            if flip: a,b,y=[x[...,::-1] for x in (a,b,y)]
        return (torch.from_numpy(a.copy()),torch.from_numpy(b.copy()),torch.from_numpy(y.copy()),
                torch.tensor(row['label_status']=='reviewed'))


def numerator(logits,y,pos_weight):
    valid=y!=255
    target=(y==1).float()
    loss=F.binary_cross_entropy_with_logits(logits.float(),target,reduction='none',pos_weight=pos_weight)
    return (loss*valid).sum()


def metrics_from(v):
    loss,n,tp,fp,fn,tn=v.tolist()
    if n==0: return None
    return dict(loss=loss/n,pixels=int(n),tp=int(tp),fp=int(fp),fn=int(fn),tn=int(tn),
                iou=tp/(tp+fp+fn) if tp+fp+fn else None,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                precision=tp/(tp+fp) if tp+fp else None,
                recall=tp/(tp+fn) if tp+fn else None)


@torch.no_grad()
def evaluate(model,loader,device,amp,pos_weight,world):
    model.eval();totals=torch.zeros(3,6,dtype=torch.float64,device=device)
    for a,b,y,reviewed in loader:
        a,b,y,reviewed=[x.to(device) for x in (a,b,y,reviewed)]
        with amp(): logits=model(a,b)
        valid=y!=255;pred=logits>=0;truth=y==1
        per=F.binary_cross_entropy_with_logits(logits.float(),truth.float(),reduction='none',pos_weight=pos_weight)
        for idx,select in enumerate((torch.ones_like(reviewed),reviewed,~reviewed)):
            mask=valid&select[:,None,None]
            totals[idx]+=torch.stack(((per*mask).sum(),mask.sum(),(pred&truth&mask).sum(),
                                     (pred&~truth&mask).sum(),(~pred&truth&mask).sum(),(~pred&~truth&mask).sum())).double()
    if world>1: dist.all_reduce(totals)
    return {name:metrics_from(v) for name,v in zip(('all','reviewed','pseudo'),totals.cpu())}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True,help='Dataset directory containing dataset_manifest.csv and stats.json')
    init=p.add_mutually_exclusive_group(required=True)
    init.add_argument('--init-mae',type=Path)
    init.add_argument('--init-change',type=Path,help='New phase from a change checkpoint; fresh optimizer')
    init.add_argument('--resume',type=Path,help='Resume at last completed epoch, same data and hyperparameters')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=10)
    p.add_argument('--train-blocks',type=int,default=0,help='0 freezes encoder; N unfreezes last N blocks and norm')
    p.add_argument('--head-width',type=int,default=128)
    p.add_argument('--lr',type=float,default=1e-4,help='New head peak LR')
    p.add_argument('--encoder-lr',type=float,default=1e-6)
    p.add_argument('--min-lr-ratio',type=float,default=.1)
    p.add_argument('--warmup-epochs',type=int,default=1)
    p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--accum-steps',type=int,default=8)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--amp',choices=['bf16','fp16','off'],default='bf16')
    p.add_argument('--pos-weight',type=float,default=1.,help='BCE weight of changed class; fixed across train/val')
    p.add_argument('--weight-decay',type=float,default=.01)
    p.add_argument('--clip-grad',type=float,default=1.)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--log-every',type=int,default=20)
    p.add_argument('--eval-only',action='store_true')
    p.add_argument('--eval-split',choices=['val','test'],default='val')
    a=p.parse_args()
    if min(a.epochs,a.batch_size,a.accum_steps,a.log_every)<1 or a.workers<0: p.error('Invalid sizes')
    if not 0<=a.warmup_epochs<a.epochs or not 0<a.min_lr_ratio<=1: p.error('Invalid schedule')
    if min(a.lr,a.encoder_lr,a.pos_weight,a.clip_grad)<=0 or a.weight_decay<0: p.error('Invalid optimizer/loss options')
    if not a.eval_only and a.eval_split!='val': p.error('Test split is only allowed with --eval-only')
    rank=int(os.environ.get('RANK',0));world=int(os.environ.get('WORLD_SIZE',1));local=int(os.environ.get('LOCAL_RANK',0))
    device=torch.device('cuda',local) if torch.cuda.is_available() else torch.device('cpu')
    if device.type=='cuda': torch.cuda.set_device(device)
    if world>1: dist.init_process_group('nccl' if device.type=='cuda' else 'gloo')
    try:
        torch.manual_seed(a.seed);random.seed(a.seed);np.random.seed(a.seed)
        if a.amp!='off' and device.type!='cuda': p.error('Use --amp off on CPU')
        if a.amp=='bf16' and not torch.cuda.is_bf16_supported(): p.error('bf16 unsupported; use fp16')
        amp=(lambda:contextlib.nullcontext()) if a.amp=='off' else (lambda:torch.autocast('cuda',dtype=torch.bfloat16 if a.amp=='bf16' else torch.float16))
        path=a.data/'dataset_manifest.csv';rows=read_rows(path)
        stats=json.loads((a.data/'stats.json').read_text())
        config=json.loads((a.data/'build_config.json').read_text())
        size=config['size'];fingerprint=data_fingerprint(path,rows)
        source=a.resume or a.init_change or a.init_mae
        payload=load_payload(source)
        check_stats(payload['stats'],stats)
        if a.init_mae:
            if payload['stats']['tile_size']!=size: raise ValueError('Use dataset --size equal to MAE tile_size; no hidden positional interpolation')
            arch=payload['args']['arch'];width=a.head_width
        else:
            if payload.get('format')!='nisar_change_v1': raise ValueError('Expected change-segmentation checkpoint')
            arch=payload['arch'];width=payload['head_width']
            if payload['size']!=size or width!=a.head_width: raise ValueError('Model size/head width mismatch')
        model=ChangeModel(size,arch,width)
        if a.init_mae: load_mae_encoder(model,payload)
        else: model.load_state_dict(payload['model'],strict=True)
        model.set_trainable(a.train_blocks);model.to(device)
        val=Tiles([r for r in rows if r['split']==a.eval_split],stats,size)
        if not len(val): raise ValueError(f'No supervised pixels in {a.eval_split}')
        kw=dict(batch_size=a.batch_size,num_workers=a.workers,pin_memory=device.type=='cuda')
        vloader=DataLoader(val,sampler=list(range(rank,len(val),world)),**kw)
        pos_weight=torch.tensor(a.pos_weight,device=device)
        a.out.mkdir(parents=True,exist_ok=True)
        if a.eval_only:
            if a.init_mae: raise ValueError('Evaluate a trained change checkpoint with --init-change')
            result=evaluate(model,vloader,device,amp,pos_weight,world)
            if rank==0:
                print(json.dumps(result,indent=2));(a.out/f'evaluation_{a.eval_split}.json').write_text(json.dumps(result,indent=2))
            return
        train=Tiles([r for r in rows if r['split']=='train'],stats,size,True,a.seed)
        if not len(train) or min(train.counts[:2])==0: raise ValueError('Training needs labeled unchanged AND changed pixels')
        sampler=DistributedSampler(train,num_replicas=world,rank=rank,shuffle=True,seed=a.seed)
        loader=DataLoader(train,sampler=sampler,**kw)
        groups=[]
        for enc in (True,False):
            params=[q for name,q in model.named_parameters() if q.requires_grad and name.startswith('encoder.')==enc]
            if params: groups.append(dict(params=params,lr=a.encoder_lr if enc else a.lr,base_lr=a.encoder_lr if enc else a.lr))
        optimizer=torch.optim.AdamW(groups,weight_decay=a.weight_decay,foreach=False)
        scaler=torch.amp.GradScaler('cuda',enabled=a.amp=='fp16')
        steps_per_epoch=math.ceil(len(loader)/a.accum_steps)
        selection='reviewed' if any(r['label_status']=='reviewed' for r in val.rows) else 'all'
        start=0;step=0;best=float('inf')
        saved_args={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
        locked=('epochs','train_blocks','head_width','lr','encoder_lr','min_lr_ratio','warmup_epochs',
                'batch_size','accum_steps','amp','pos_weight','weight_decay','clip_grad','seed')
        if a.resume:
            if payload['fingerprint']!=fingerprint or payload['world_size']!=world: raise ValueError('Resume requires same dataset and world size; use --init-change for new data')
            for k in locked:
                if payload['args'][k]!=saved_args[k]: raise ValueError(f'Resume must preserve {k}')
            optimizer.load_state_dict(payload['optimizer']);scaler.load_state_dict(payload['scaler'])
            start=payload['epoch']+1;step=payload['step'];best=payload['best']
        elif any((a.out/n).exists() for n in ('last.pt','best.pt','metrics.jsonl')):
            raise ValueError('Existing output; use a new --out or --resume')
        del payload
        wrapped=DDP(model,device_ids=[local] if device.type=='cuda' else None,broadcast_buffers=False) if world>1 else model
        if rank==0:
            print(f'Train tiles: {len(train)}; validation tiles: {len(val)}; train counts 0/1/255: {train.counts.tolist()}',flush=True)
            print(f'Trainable parameters: {sum(q.numel() for q in model.parameters() if q.requires_grad):,}; encoder blocks: {a.train_blocks}',flush=True)
            print(f'Best checkpoint uses {selection} validation BCE; pseudo-label agreement is not independent change accuracy.',flush=True)
            (a.out/'config.json').write_text(json.dumps(saved_args,indent=2))
        initial=evaluate(model,vloader,device,amp,pos_weight,world)
        if rank==0: (a.out/'initial_validation.json').write_text(json.dumps(initial,indent=2))
        def snapshot(epoch):
            return dict(format='nisar_change_v1',model=model.state_dict(),optimizer=optimizer.state_dict(),scaler=scaler.state_dict(),
                        epoch=epoch,step=step,best=best,args=saved_args,stats=stats,arch=arch,size=size,head_width=width,
                        fingerprint=fingerprint,world_size=world,selection=selection,source_checkpoint=str(source.resolve()))
        if not a.resume:
            best=initial[selection]['loss']
            if rank==0: atomic_save(snapshot(-1),a.out/'best.pt')
        for epoch in range(start,a.epochs):
            begin=time.monotonic();sampler.set_epoch(epoch);train.epoch=epoch
            torch.manual_seed(a.seed+epoch*world+rank)
            wrapped.train();optimizer.zero_grad(set_to_none=True)
            total=torch.zeros(2,dtype=torch.float64,device=device)
            iterator=iter(loader)
            while True:
                batches=list(itertools.islice(iterator,a.accum_steps))
                if not batches: break
                # One pixel-weighted denominator across ALL accumulation microbatches and ranks.
                count=torch.tensor(sum(int((batch[2]!=255).sum()) for batch in batches),device=device,dtype=torch.float64)
                if world>1: dist.all_reduce(count)
                if count==0: continue
                warm=a.warmup_epochs*steps_per_epoch;end=a.epochs*steps_per_epoch
                factor=(step+1)/warm if warm and step<warm else a.min_lr_ratio+(1-a.min_lr_ratio)*.5*(1+math.cos(math.pi*min(1,(step-warm)/max(1,end-warm-1))))
                for group in optimizer.param_groups: group['lr']=group['base_lr']*factor
                for j,(x0,x1,y,_) in enumerate(batches):
                    x0,x1,y=[x.to(device,non_blocking=True) for x in (x0,x1,y)]
                    sync=wrapped.no_sync() if world>1 and j<len(batches)-1 else contextlib.nullcontext()
                    with sync:
                        with amp(): logits=wrapped(x0,x1)
                        num=numerator(logits,y,pos_weight)
                        if not torch.isfinite(num): raise FloatingPointError('Nonfinite training loss')
                        scaler.scale(num*(world/count.float())).backward()
                    total+=torch.stack((num.detach().double(),(y!=255).sum().double()))
                scaler.unscale_(optimizer)
                grad=torch.nn.utils.clip_grad_norm_([q for q in model.parameters() if q.requires_grad],a.clip_grad)
                if a.amp!='fp16' and not torch.isfinite(grad): raise FloatingPointError('Nonfinite gradients')
                scaler.step(optimizer);scaler.update();optimizer.zero_grad(set_to_none=True);step+=1
                if rank==0 and step%a.log_every==0: print(f'epoch={epoch+1} step={step} head_lr={a.lr*factor:.3g}',flush=True)
            if world>1: dist.all_reduce(total)
            result=evaluate(model,vloader,device,amp,pos_weight,world)
            value=result[selection]['loss']
            if not math.isfinite(value): raise FloatingPointError('Nonfinite validation loss')
            improved=value<best;best=min(best,value)
            if rank==0:
                record=dict(epoch=epoch+1,step=step,train_loss=(total[0]/total[1]).item(),validation=result,
                            best=best,seconds=time.monotonic()-begin)
                with (a.out/'metrics.jsonl').open('a') as f: f.write(json.dumps(record,allow_nan=False)+'\n')
                atomic_save(snapshot(epoch),a.out/'last.pt')
                if improved: atomic_save(snapshot(epoch),a.out/'best.pt')
                print(json.dumps(record),flush=True)
            if world>1: dist.barrier()
    finally:
        if world>1 and dist.is_initialized(): dist.destroy_process_group()

if __name__=='__main__': main()
