"""Synthetic regression tests: no NISAR downloads or large checkpoint required."""
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import h5py
import numpy as np
import rasterio
import torch
from affine import Affine
from build_change_dataset import read_window
from change_model import ChangeModel,load_mae_encoder
from finetune_change import numerator

ROOT=Path(__file__).resolve().parent


def command(*args,good=True):
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    result=subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,env=env,capture_output=True,text=True,timeout=120)
    if good and result.returncode: raise AssertionError(result.stdout+'\n'+result.stderr)
    if not good and result.returncode==0: raise AssertionError('Expected rejection')
    return result


def scene(path,x0,changed=False,offset=0):
    h,w=61,65
    with h5py.File(path,'w') as f:
        g=f.create_group('/science/LSAR/GCOV/grids/frequencyA')
        g['xCoordinates']=x0+offset+10+np.arange(w)*20
        g['yCoordinates']=1000000-10-np.arange(h)*20
        g['xCoordinateSpacing']=20.;g['yCoordinateSpacing']=-20.;g['projection']=32622
        for name in ('HHHH','HVHV'):
            data=np.full((h,w),.1,np.float32)
            if changed: data[17:49,17:49]=.4
            data[:2,:]=np.nan
            g.create_dataset(name,data=data,fillvalue=np.nan)


class PipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.tmp=tempfile.TemporaryDirectory();cls.base=Path(cls.tmp.name)
        cls.stats=dict(frequency='A',channels=['HHHH','HVHV'],spacing_m=20.,db_clip=[-30.,10.],
                       mean=[-10.,-10.],std=[5.,5.],tile_size=28)
        (cls.base/'stats.json').write_text(json.dumps(cls.stats))
        rows=[]
        for area,x in [('train_area',300000),('val_area',700000),('new_area',900000)]:
            t0=cls.base/f'{area}0.h5';t1=cls.base/f'{area}1.h5'
            scene(t0,x);scene(t1,x,True)
            rows.append(dict(pair_id=area,area=area,split='val' if area=='val_area' else 'train',t0_path=str(t0),t1_path=str(t1)))
        cls.rows=rows
        cls.manifest=cls.base/'pairs.csv'
        with cls.manifest.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows[:2])
        cls.data=cls.base/'data'
        cls.build=['build_change_dataset.py','--manifest',cls.manifest,'--stats',cls.base/'stats.json','--out',cls.data,'--min-pixels',4]
        command(*cls.build)
    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()

    def test_01_full_coverage_padding_and_append(self):
        mf=self.data/'dataset_manifest.csv'
        before=mf.read_bytes()
        command(*self.build)
        self.assertEqual(before,mf.read_bytes())
        with mf.open() as f: rows=list(csv.DictReader(f))
        self.assertEqual(len(rows),18)  # ceil(61/28)*ceil(65/28) for each pair
        # Simulate an interruption after tile commit but before CSV checkpoint.
        with mf.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows[:-1])
        command(*self.build)
        with mf.open() as f: recovered=list(csv.DictReader(f))
        self.assertEqual(recovered,rows)
        coverage=np.zeros((84,84),int)
        for r in rows[:9]:
            rr=int(r['crop_id'][1:7]);cc=int(r['crop_id'][9:])
            with rasterio.open(self.data/r['valid']) as ds: v=ds.read(1)
            coverage[rr:rr+28,cc:cc+28]+=v
        self.assertTrue((coverage[2:61,:65]==1).all())
        self.assertFalse(coverage[61:].any());self.assertFalse(coverage[:,65:].any())
        corrected=self.data/rows[0]['label']
        with rasterio.open(corrected,'r+') as ds:
            y=ds.read(1);y[10,10]=1;ds.write(y,1)
        marker=corrected.read_bytes()
        with self.manifest.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(self.rows[0]));w.writeheader();w.writerows(self.rows)
        command(*self.build)
        with mf.open() as f: after=list(csv.DictReader(f))
        self.assertEqual(after[:18],rows);self.assertEqual(len(after),27)
        self.assertEqual(corrected.read_bytes(),marker)
        # Changed sources/settings may not silently overwrite corrections.
        command(*self.build,'--high-db',4,good=False)

    def test_02_linear_power_average_and_validity(self):
        with h5py.File(self.base/'read.h5','w') as f:
            g=f.create_group('g')
            for ch in ('HHHH','HVHV'): g.create_dataset(ch,data=np.array([[1.,9.],[1.,9.]],np.float32),fillvalue=np.nan)
            x,v=read_window(g,(2,2),['HHHH','HVHV'],0,0,1,1,2,2,[-30,20])
            self.assertTrue(v[0,0]);self.assertAlmostEqual(float(x[0,0,0]),10*np.log10(5),places=5)
            x,v=read_window(g,(2,2),['HHHH','HVHV'],-2,0,2,1,2,2,[-30,20])
            self.assertFalse(v[0,0]);self.assertTrue(v[1,0])

    def test_03_model_symmetry_loss_and_loading(self):
        model=ChangeModel(28,'tiny',16);model.set_trainable(0)
        enc={k:v.clone() for k,v in model.encoder.state_dict().items()}
        enc['decoder_pred.weight']=torch.randn(3,3)
        payload=dict(model=enc,stats=self.stats,args={'arch':'tiny'})
        load_mae_encoder(model,payload)
        bad=dict(enc);bad.pop('norm.weight')
        with self.assertRaises(RuntimeError):load_mae_encoder(model,dict(payload,model=bad))
        a,b=torch.randn(1,2,28,28),torch.randn(1,2,28,28)
        model.eval()
        self.assertTrue(torch.allclose(model(a,b),model(b,a),atol=1e-6))
        logits=torch.zeros(1,2,2,requires_grad=True);y=torch.tensor([[[0,1],[255,255]]])
        loss=numerator(logits,y,torch.tensor(1.));loss.backward()
        self.assertTrue((logits.grad[0,1]==0).all())
        model.train();model(a,b).mean().backward()
        self.assertTrue(all(p.grad is None for p in model.encoder.parameters()))
        self.assertIsNotNone(model.refine[-1].weight.grad)
        model.zero_grad();model.set_trainable(1);model(a,b).mean().backward()
        self.assertIsNotNone(model.encoder.blocks[-1].attn.qkv.weight.grad)

    def test_04_train_resume_predict(self):
        mae=ChangeModel(28,'tiny',16)
        ckpt=self.base/'mae.pt'
        torch.save(dict(model=mae.encoder.state_dict(),stats=self.stats,args={'arch':'tiny'}),ckpt)
        out=self.base/'run'
        common=['--data',self.data,'--out',out,'--epochs',2,'--head-width',16,'--workers',0,'--amp','off','--batch-size',2,'--accum-steps',2]
        command('finetune_change.py',*common,'--init-mae',ckpt)
        command('finetune_change.py',*common,'--resume',out/'last.pt')
        command('finetune_change.py','--data',self.data,'--out',self.base/'last_block_run',
                '--init-change',out/'best.pt','--epochs',2,'--head-width',16,'--train-blocks',1,
                '--workers',0,'--amp','off','--batch-size',2,'--accum-steps',2)
        command('predict_change.py','--data',self.data,'--checkpoint',out/'best.pt','--out',self.base/'pred','--amp','off')
        with (self.base/'pred/prediction_manifest.csv').open() as f: predictions=list(csv.DictReader(f))
        self.assertEqual(len(predictions),9)
        with rasterio.open(predictions[0]['label']) as ds: self.assertTrue(np.isin(ds.read(1),[0,1,255]).all())
        command('stitch_nisar_labels.py','--manifest',self.base/'pred/prediction_manifest.csv','--out',self.base/'mosaic')
        with rasterio.open(self.base/'mosaic/val_area/labels_stitched.tif') as ds:
            self.assertEqual(ds.shape,(84,84))

    def test_05_fractional_and_split_rejection(self):
        scene(self.base/'fractional.h5',300000,True,offset=5)
        row=dict(self.rows[0],t1_path=str(self.base/'fractional.h5'))
        manifest=self.base/'bad.csv'
        with manifest.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(row));w.writeheader();w.writerow(row)
        command('build_change_dataset.py','--manifest',manifest,'--stats',self.base/'stats.json','--out',self.base/'bad','--dry-run',good=False)
        row=dict(self.rows[0],pair_id='leak',area='different_name',split='val')
        with manifest.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(row));w.writeheader();w.writerows([self.rows[0],row])
        command('build_change_dataset.py','--manifest',manifest,'--stats',self.base/'stats.json','--out',self.base/'bad','--dry-run',good=False)

    @unittest.skipUnless(os.environ.get('NISAR_TEST_DDP')=='1', 'Set NISAR_TEST_DDP=1 for local two-rank test')
    def test_06_two_rank_cpu_ddp(self):
        out=self.base/'ddp'
        command('-m','torch.distributed.run','--master_addr=127.0.0.1','--master_port=29591','--nproc_per_node=2','finetune_change.py',
                '--data',self.data,'--out',out,'--epochs',2,'--head-width',16,'--workers',0,
                '--amp','off','--batch-size',1,'--accum-steps',2,'--init-mae',self.base/'mae.pt')
        single=torch.load(self.base/'run/last.pt',weights_only=True,map_location='cpu')
        ddp=torch.load(out/'last.pt',weights_only=True,map_location='cpu')
        self.assertEqual(single['step'],ddp['step'])
        # Pixel-weighted accumulation should agree across different rank/batch layouts.
        for name,value in single['model'].items():
            self.assertTrue(torch.allclose(value,ddp['model'][name],atol=2e-5,rtol=1e-4),name)

if __name__=='__main__': unittest.main(verbosity=2)
