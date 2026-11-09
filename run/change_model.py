"""SARMAE-compatible ViT encoder and a symmetric, dense bi-temporal change head.
Encoder parameter names match the supplied nisar_mae.py SARMAE implementation.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


class PatchEmbed(nn.Module):
    def __init__(self,dim,patch_size):
        super().__init__();self.proj=nn.Conv2d(2,dim,patch_size,stride=patch_size)
    def forward(self,x): return self.proj(x).flatten(2).transpose(1,2)


class Attention(nn.Module):
    def __init__(self,dim,heads):
        super().__init__();self.heads=heads
        self.qkv=nn.Linear(dim,3*dim);self.proj=nn.Linear(dim,dim)
    def forward(self,x):
        b,n,d=x.shape
        q,k,v=self.qkv(x).reshape(b,n,3,self.heads,d//self.heads).permute(2,0,3,1,4).unbind(0)
        y=F.scaled_dot_product_attention(q,k,v,dropout_p=0.)
        return self.proj(y.transpose(1,2).reshape(b,n,d))


class MLP(nn.Module):
    def __init__(self,dim):
        super().__init__();self.fc1=nn.Linear(dim,4*dim);self.fc2=nn.Linear(4*dim,dim)
    def forward(self,x): return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self,dim,heads):
        super().__init__();self.norm1=nn.LayerNorm(dim,eps=1e-6);self.norm2=nn.LayerNorm(dim,eps=1e-6)
        self.attn=Attention(dim,heads);self.mlp=MLP(dim)
    def forward(self,x):
        x=x+self.attn(self.norm1(x));return x+self.mlp(self.norm2(x))


class Encoder(nn.Module):
    def __init__(self,size,arch,grad_checkpoint=True):
        super().__init__()
        dim,depth,heads={'huge':(1280,32,16),'tiny':(64,2,4)}[arch]
        if size%14: raise ValueError('Input size must be divisible by patch size 14')
        self.image_size=size;self.patch_size=14;self.dim=dim;self.grad_checkpoint=grad_checkpoint
        self.patch_embed=PatchEmbed(dim,14)
        self.cls_token=nn.Parameter(torch.zeros(1,1,dim))
        self.pos_embed=nn.Parameter(torch.zeros(1,(size//14)**2+1,dim),requires_grad=False)
        self.blocks=nn.ModuleList([Block(dim,heads) for _ in range(depth)])
        self.norm=nn.LayerNorm(dim,eps=1e-6)

    def forward(self,x):
        if x.shape[1:]!=(2,self.image_size,self.image_size): raise ValueError('Input/checkpoint tile-size mismatch')
        z=self.patch_embed(x)+self.pos_embed[:,1:]
        cls=(self.cls_token+self.pos_embed[:,:1]).expand(x.shape[0],-1,-1)
        z=torch.cat((cls,z),1)
        for block in self.blocks:
            if self.training and self.grad_checkpoint and any(p.requires_grad for p in block.parameters()):
                z=checkpoint(block,z,use_reentrant=False)
            else: z=block(z)
        z=self.norm(z)[:,1:]
        side=self.image_size//14
        return z.transpose(1,2).reshape(x.shape[0],self.dim,side,side)


def conv(cin,cout):
    return nn.Sequential(nn.Conv2d(cin,cout,3,padding=1),nn.GroupNorm(8,cout),nn.GELU())


class ChangeModel(nn.Module):
    def __init__(self,size,arch='huge',width=128,grad_checkpoint=True):
        super().__init__()
        if width<16 or width%8: raise ValueError('Head width must be >=16 and divisible by 8')
        self.encoder=Encoder(size,arch,grad_checkpoint)
        # Mean and absolute difference make prediction invariant to temporal order.
        self.fuse=nn.Sequential(nn.Conv2d(self.encoder.dim*2,width,1),nn.GroupNorm(8,width),nn.GELU(),conv(width,width))
        self.local=nn.Sequential(conv(4,32),conv(32,32))
        self.refine=nn.Sequential(conv(width+32,64),conv(64,32),nn.Conv2d(32,1,1))

    def forward(self,a,b):
        f0=self.encoder(a);f1=self.encoder(b)
        z=self.fuse(torch.cat(((f0+f1)*.5,(f1-f0).abs()),1))
        z=F.interpolate(z,size=a.shape[-2:],mode='bilinear',align_corners=False)
        skip=self.local(torch.cat(((a+b)*.5,(b-a).abs()),1))
        return self.refine(torch.cat((z,skip),1))[:,0]

    def set_trainable(self,last_blocks):
        depth=len(self.encoder.blocks)
        if not 0<=last_blocks<=depth: raise ValueError(f'train-blocks must be in [0,{depth}]')
        for p in self.encoder.parameters(): p.requires_grad_(False)
        if last_blocks:
            for block in self.encoder.blocks[-last_blocks:]:
                for p in block.parameters(): p.requires_grad_(True)
            for p in self.encoder.norm.parameters(): p.requires_grad_(True)
        # Patch embedding, CLS, earlier blocks stay frozen even in partial tuning.


def load_mae_encoder(model,payload):
    if 'model' not in payload or 'stats' not in payload or 'args' not in payload:
        raise ValueError('Expected SARMAE checkpoint containing model, stats and args')
    state=payload['model']
    if all(k.startswith('module.') for k in state): state={k[7:]:v for k,v in state.items()}
    encoder={k:v for k,v in state.items() if not k.startswith('decoder_') and k!='mask_token'}
    # Strict matching: never silently train with missing/random encoder weights.
    model.encoder.load_state_dict(encoder,strict=True)


def load_payload(path):
    return torch.load(path,map_location='cpu',weights_only=True,mmap=True)
