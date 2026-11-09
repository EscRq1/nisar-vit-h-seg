#!/usr/bin/env python3
"""Review, train, and classify tile-level SAR change candidates."""
import argparse, csv, hashlib, json, random
from pathlib import Path
import numpy as np
import rasterio
import torch
from torch import nn
from scipy.ndimage import label as connected
from change_model import ChangeModel, load_payload

FIELDS = ['region_id','pair_id','area','split','crop_id','t0','t1','valid','mask','preview','label','evidence','notes']

def rows_read(path):
    with Path(path).open(newline='') as f: return list(csv.DictReader(f))

def write_csv(path, rows, fields):
    with Path(path).open('x', newline='') as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

def resolved(base, value):
    p=Path(value); return (p if p.is_absolute() else base/p).resolve()

def image_data(row):
    arrays=[]; grid=None
    for key in ('t0','t1','valid','mask'):
        with rasterio.open(row[key]) as s:
            g=(s.crs,s.transform,s.width,s.height)
            if grid is not None and g!=grid: raise ValueError(f'Grid mismatch: {row[key]}')
            grid=g; arrays.append(s.read())
    a,b,v,m=arrays
    valid=(v[0]==1)&np.isfinite(a).all(0)&np.isfinite(b).all(0)
    mask=(m[0]==1)&valid
    if not mask.any(): raise ValueError(f'Empty valid region: {row["region_id"]}')
    return a,b,valid,mask

def prepare(a):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    src=Path(a.manifest).resolve(); inputs=rows_read(src)
    selected=[r for r in inputs if not a.pair_id or r['pair_id'] in a.pair_id]
    if not selected: raise ValueError('No matching pairs')
    out=Path(a.out).resolve(); out.mkdir(parents=True,exist_ok=False)
    rows=[]
    for row in selected:
        for key in ('t0','t1','valid','label'): row[key]=str(resolved(src.parent,row[key]))
        with rasterio.open(row['label']) as s:
            arr=s.read(1); profile=s.profile
        components,n=connected(arr==1, structure=np.ones((3,3)))
        for i in range(1,n+1):
            m=components==i
            if m.sum()<a.min_pixels: continue
            rid=f"{row['pair_id']}_{row['crop_id']}_{i:04d}"
            dest=out/rid; dest.mkdir()
            profile.update(count=1,dtype='uint8',nodata=255,compress='deflate')
            with rasterio.open(dest/'region.tif','w',**profile) as ds: ds.write(m.astype('uint8'),1)
            item=dict(region_id=rid,pair_id=row['pair_id'],area=row['area'],split=row['split'],crop_id=row['crop_id'],t0=row['t0'],t1=row['t1'],valid=row['valid'],mask=str(dest/'region.tif'),preview=str(dest/'preview.png'),label='',evidence='',notes='')
            x,y,v,m=image_data(item)
            fig,axes=plt.subplots(1,3,figsize=(10,3.5))
            for ax,z,title in zip(axes,(x,y,y),('T0 HHHH','T1 HHHH','T1 + region')):
                ax.imshow(np.where(v,z[0],np.nan),cmap='gray',vmin=-30,vmax=0); ax.set_title(title); ax.axis('off')
            axes[2].imshow(np.ma.masked_where(~m,m),cmap='autumn',alpha=.55,vmin=0,vmax=1)
            fig.suptitle(rid,fontsize=8); fig.tight_layout(); fig.savefig(dest/'preview.png',dpi=110); plt.close(fig)
            rows.append(item)
            if a.max_regions and len(rows)>=a.max_regions: break
        if len(rows)%100==0 and rows: print('Prepared',len(rows),flush=True)
        if a.max_regions and len(rows)>=a.max_regions: break
    if not rows: raise ValueError('No components above minimum size')
    write_csv(out/'regions.csv',rows,FIELDS); print('Regions:',len(rows),'manifest:',out/'regions.csv')

def checkpoint_hash(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()

def encoder_load(path,device):
    saved=load_payload(path)
    if saved.get('format')!='nisar_change_v1': raise ValueError('Expected fine-tuned change checkpoint (nisar_change_v1)')
    m=ChangeModel(saved['size'],saved['arch'],saved['head_width'],grad_checkpoint=False)
    m.load_state_dict(saved['model'],strict=True)
    enc=m.encoder.to(device).eval()
    for p in enc.parameters(): p.requires_grad_(False)
    stats=saved['stats']; size=saved['size']; del m,saved
    return enc,stats,size

def feature(row,enc,stats,size,device):
    x,y,valid,mask=image_data(row)
    if x.shape!=(2,size,size) or y.shape!=x.shape: raise ValueError('Tile shape incompatible with checkpoint')
    mean=np.asarray(stats['mean'],np.float32)[:,None,None]; std=np.asarray(stats['std'],np.float32)[:,None,None]
    a,b=[torch.from_numpy(np.where(valid[None],(z-mean)/std,0).astype(np.float32))[None].to(device) for z in (x,y)]
    with torch.inference_mode():
        f0=enc(a).float(); f1=enc(b).float()
        weights=torch.nn.functional.adaptive_avg_pool2d(torch.from_numpy(mask.astype(np.float32))[None,None].to(device),f0.shape[-2:])
        weights=weights / weights.sum().clamp_min(1e-12)
        p0=(f0*weights).sum((-1,-2)); p1=(f1*weights).sum((-1,-2))
        return torch.cat((p0,p1,p1-p0),dim=1)[0].cpu()

class Head(nn.Module):
    def __init__(self,dim,n):
        super().__init__(); self.net=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,256),nn.GELU(),nn.Dropout(.2),nn.Linear(256,n))
    def forward(self,x): return self.net(x)

def dataset(path):
    path=Path(path).resolve(); rows=rows_read(path)
    for r in rows:
        for k in ('t0','t1','valid','mask'): r[k]=str(resolved(path.parent,r[k]))
    return rows

def train(a):
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    rows=[r for r in dataset(a.regions) if r.get('label','').strip() and r['label'].strip().lower() not in ('unknown','skip','uncertain')]
    if not rows: raise ValueError('No reviewed labeled regions')
    if any(r['split'] not in ('train','val') for r in rows): raise ValueError('Reviewed rows must be train or val')
    for key in ('area','pair_id'):
        groups={}
        for r in rows:
            if not r.get(key): raise ValueError(f'Missing {key}')
            if r[key] in groups and groups[r[key]]!=r['split']: raise ValueError(f'{key} crosses splits: {r[key]}')
            groups[r[key]]=r['split']
    tr=[r for r in rows if r['split']=='train']; va=[r for r in rows if r['split']=='val']
    classes=sorted({r['label'].strip() for r in tr})
    if len(classes)<2 or not va: raise ValueError('Need >=2 training classes and a separate validation set')
    if set(r['label'].strip() for r in va)-set(classes): raise ValueError('Validation labels absent from training')
    device=torch.device(a.device); print('Loading encoder',flush=True); enc,stats,size=encoder_load(a.checkpoint,device)
    feats=[]
    for i,r in enumerate(rows):
        feats.append(feature(r,enc,stats,size,device))
        if (i+1)%25==0: print('Encoded',i+1,'/',len(rows),flush=True)
    del enc
    X=torch.stack(feats).to(device); y=torch.tensor([classes.index(r['label'].strip()) for r in rows],device=device)
    ti=torch.tensor([i for i,r in enumerate(rows) if r['split']=='train'],device=device); vi=torch.tensor([i for i,r in enumerate(rows) if r['split']=='val'],device=device)
    head=Head(X.shape[1],len(classes)).to(device); opt=torch.optim.AdamW(head.parameters(),lr=a.lr,weight_decay=.01)
    out=Path(a.out); out.mkdir(parents=True,exist_ok=False); best=float('inf'); digest=checkpoint_hash(a.checkpoint)
    for epoch in range(1,a.epochs+1):
        head.train(); order=ti[torch.randperm(len(ti),device=device)]
        for batch in order.split(a.batch_size):
            loss=nn.functional.cross_entropy(head(X[batch]),y[batch]);opt.zero_grad();loss.backward();opt.step()
        head.eval()
        with torch.no_grad():
            logits=head(X[vi]); vloss=nn.functional.cross_entropy(logits,y[vi]).item(); pred=logits.argmax(1); matrix=torch.zeros((len(classes),len(classes)),dtype=torch.long,device=device)
            for truth,guess in zip(y[vi],pred): matrix[truth,guess]+=1
        print(json.dumps({'epoch':epoch,'val_loss':vloss,'val_accuracy':(pred==y[vi]).float().mean().item(),'classes':classes,'confusion_matrix':matrix.cpu().tolist()}),flush=True)
        if vloss<best:
            best=vloss; torch.save({'format':'nisar_region_v1','head':head.cpu().state_dict(),'classes':classes,'dim':X.shape[1],'encoder_sha256':digest,'size':size,'stats':stats},out/'best.pt');head.to(device)
    print('Saved',out/'best.pt')

def predict(a):
    device=torch.device(a.device); saved=torch.load(a.model,map_location='cpu',weights_only=True)
    if saved.get('format')!='nisar_region_v1': raise ValueError('Invalid classifier checkpoint')
    if checkpoint_hash(a.checkpoint)!=saved['encoder_sha256']: raise ValueError('Change checkpoint differs from classifier training checkpoint')
    enc,stats,size=encoder_load(a.checkpoint,device); head=Head(saved['dim'],len(saved['classes'])).to(device);head.load_state_dict(saved['head']);head.eval()
    rows=dataset(a.regions); out=[]
    for i,r in enumerate(rows):
        x=feature(r,enc,stats,size,device).to(device)
        with torch.inference_mode(): scores=head(x[None]).softmax(-1)[0].cpu().numpy()
        idx=int(scores.argmax()); score=float(scores[idx]); r['predicted_class']=saved['classes'][idx] if score>=a.reject_below else 'unknown';r['top_score']=score;r['class_scores']=json.dumps(dict(zip(saved['classes'],map(float,scores))))
        out.append(r)
        if (i+1)%25==0: print('Classified',i+1,'/',len(rows),flush=True)
    if not out: raise ValueError('Empty region manifest')
    write_csv(a.out,out,list(out[0]));print('Saved',a.out,'Scores are uncalibrated; unknown threshold is a heuristic.')

def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='cmd',required=True)
    q=sub.add_parser('prepare');q.add_argument('--manifest',required=True);q.add_argument('--out',required=True);q.add_argument('--pair-id',nargs='+');q.add_argument('--min-pixels',type=int,default=10);q.add_argument('--max-regions',type=int,default=0);q.set_defaults(fn=prepare)
    q=sub.add_parser('train');q.add_argument('--regions',required=True);q.add_argument('--checkpoint',required=True);q.add_argument('--out',required=True);q.add_argument('--epochs',type=int,default=30);q.add_argument('--batch-size',type=int,default=16);q.add_argument('--lr',type=float,default=1e-3);q.add_argument('--seed',type=int,default=42);q.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu');q.set_defaults(fn=train)
    q=sub.add_parser('predict');q.add_argument('--regions',required=True);q.add_argument('--checkpoint',required=True);q.add_argument('--model',required=True);q.add_argument('--out',required=True);q.add_argument('--reject-below',type=float,default=.7);q.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu');q.set_defaults(fn=predict)
    a=p.parse_args();a.fn(a)
if __name__=='__main__': main()
