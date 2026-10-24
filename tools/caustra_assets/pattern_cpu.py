"""CPU evaluation of the same MaterialX pattern IR used by the GPU.

The Perlin hash/gradient follows MaterialX mx_noise.glsl / Open Shading Language.
Copyright (c) 2009-2010 Sony Pictures Imageworks Inc., et al. All Rights Reserved.
Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:
* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.
* Neither the name of Sony Pictures Imageworks nor the names of its contributors
  may be used to endorse or promote products derived from this software without
  specific prior written permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
import numpy as np
from .model import AssetError, Op


def normalize(v):
    return v/np.maximum(np.linalg.norm(v,axis=-1,keepdims=True),1e-20)


def perlin(position,channels=1):
    p=np.asarray(position);cell=np.floor(p).astype(np.int64);f=p-cell
    fade=f*f*f*(f*(f*6-15)+10);result=np.zeros((len(p),channels))
    def rot(x,k): return (x<<np.uint32(k))|(x>>np.uint32(32-k))
    with np.errstate(over="ignore"):
        for z in (0,1):
            for y in (0,1):
                for x in (0,1):
                    seed=np.uint32(0xdeadbeef+12+13)
                    a=seed+(cell[:,0]+x).astype(np.uint32)
                    b=seed+(cell[:,1]+y).astype(np.uint32)
                    c=seed+(cell[:,2]+z).astype(np.uint32)
                    c=(c^b)-rot(b,14);a=(a^c)-rot(c,11);b=(b^a)-rot(a,25)
                    c=(c^b)-rot(b,16);a=(a^c)-rot(c,4);b=(b^a)-rot(a,14);c=(c^b)-rot(b,24)
                    h=np.stack([(c>>np.uint32(8*i))&np.uint32(15) for i in range(channels)],axis=-1)
                    dx,dy,dz=(f[:,i:i+1]-v for i,v in enumerate((x,y,z)))
                    u=np.where(h<8,dx,dy);v=np.where(h<4,dy,np.where((h==12)|(h==14),dx,dz))
                    gradient=np.where(h&1,-u,u)+np.where(h&2,-v,v)
                    weight=np.prod(np.where(np.asarray((x,y,z)),fade,1-fade),axis=-1,keepdims=True)
                    result+=gradient*weight*.9820
    return result


def sample_texture(texture,uv,flags):
    pixels=np.asarray(texture.pixels).reshape(texture.height,texture.width,4)
    # Standard MaterialX UV v=0 is the bottom; texture storage begins at top.
    coordinates=np.column_stack((uv[:,0],1-uv[:,1]))
    size=np.asarray((texture.width,texture.height));nearest=bool(flags&16)
    def fetch(indices):
        indices=indices.copy();valid=np.ones(len(indices),dtype=bool)
        for axis,mode in enumerate((flags&3,(flags>>2)&3)):
            n=size[axis];i=indices[:,axis]
            if mode==0: indices[:,axis]=i%n
            elif mode==1: indices[:,axis]=np.clip(i,0,n-1)
            elif mode==2:
                q=i%(2*n);indices[:,axis]=np.where(q<n,q,2*n-1-q)
            else:
                valid&=(i>=0)&(i<n);indices[:,axis]=np.clip(i,0,n-1)
        return pixels[indices[:,1],indices[:,0]]*valid[:,None]
    if nearest: return fetch(np.floor(coordinates*size).astype(np.int64))
    p=coordinates*size-.5;i=np.floor(p).astype(np.int64);f=p-i
    a=fetch(i)*(1-f[:,0:1])+fetch(i+[1,0])*f[:,0:1]
    b=fetch(i+[0,1])*(1-f[:,0:1])+fetch(i+[1,1])*f[:,0:1]
    return a*(1-f[:,1:2])+b*f[:,1:2]


def evaluate_program(program,textures,*,uv=None,position=None,object_position=None,normal=None,tangent=None,allow_spatial=True):
    """Return registers; spatial inputs are Nx3/Nx2, or one constant vector."""
    supplied=next((x for x in (uv,position,object_position,normal,tangent) if x is not None),None)
    count=len(supplied) if supplied is not None else 1
    registers=[]
    def vector(x):
        x=np.asarray(x,dtype=float)
        if x.ndim==1: return np.broadcast_to(x,(count,4)).copy()
        return np.pad(x,((0,0),(0,4-x.shape[1])))
    for code,value in program:
        op,a,b,c=code;v=vector(value)
        if op==Op.Constant: pass
        elif op in (Op.Texcoord,Op.Position,Op.ObjectPosition,Op.Normal,Op.Tangent):
            if not allow_spatial: raise AssetError("Homogeneous interior coefficients must be spatially constant")
            source={Op.Texcoord:uv,Op.Position:position,Op.ObjectPosition:object_position,Op.Normal:normal,Op.Tangent:tangent}[op]
            if source is None: raise AssetError("Displacement graph requires unavailable geometric input: "+str(Op(op)))
            v=vector(source)
        elif op in (Op.Add,Op.Subtract,Op.Multiply,Op.Divide,Op.Power,Op.Min,Op.Max):
            x,y=registers[a],registers[b]
            with np.errstate(all="ignore"):
                if op==Op.Add: v=x+y
                elif op==Op.Subtract: v=x-y
                elif op==Op.Multiply: v=x*y
                elif op==Op.Divide:
                    valid=abs(y)>1e-20
                    v=np.where(valid,x/np.where(valid,y,1),0)
                elif op==Op.Power: v=x**y
                elif op==Op.Min: v=np.minimum(x,y)
                else: v=np.maximum(x,y)
        elif op==Op.Mix: v=registers[a]*(1-registers[c])+registers[b]*registers[c]
        elif op==Op.Clamp: v=np.clip(registers[a],registers[b],registers[c])
        elif op==Op.Abs: v=abs(registers[a])
        elif op==Op.OneMinus: v=1-registers[a]
        elif op==Op.Sin: v=np.sin(registers[a])
        elif op==Op.Cos: v=np.cos(registers[a])
        elif op==Op.Floor: v=np.floor(registers[a])
        elif op==Op.Ceil: v=np.ceil(registers[a])
        elif op==Op.Fract: v=registers[a]-np.floor(registers[a])
        elif op in (Op.Dot,Op.Normalize,Op.Length):
            n=int(value[0]) or 3;x=registers[a][:,:n]
            if op==Op.Normalize: v=vector(normalize(x))
            else:
                q=np.sum(x*registers[b][:,:n],axis=-1) if op==Op.Dot else np.linalg.norm(x,axis=-1)
                v=np.repeat(q[:,None],4,axis=1)
        elif op==Op.Cross: v=vector(np.cross(registers[a][:,:3],registers[b][:,:3]))
        elif op==Op.Combine: v=np.column_stack((registers[a][:,0],registers[b][:,0],registers[c][:,0],np.full(count,value[3])))
        elif op==Op.Swizzle: v=registers[a][:,np.asarray(value,dtype=int)]
        elif op==Op.Image: v=sample_texture(textures[b],registers[a],c)
        elif op==Op.Checker:
            p=registers[a][:,:2]*np.asarray(value[:2]);q=np.floor(p).astype(np.int64).sum(axis=-1)&1
            v=np.where(q[:,None],registers[c],registers[b])
        elif op==Op.Noise3D:
            n=3 if int(value[0])==3 else 1;q=perlin(registers[a][:,:3],n)
            noise=np.pad(q,((0,0),(0,4-n))) if n==3 else np.repeat(q,4,axis=1)
            v=noise*registers[b]+registers[c]
        elif op==Op.RgbToHsv:
            # MaterialX mx_hsv.glsl / OSL; preserve alpha for color4.
            x=registers[a];r,g,b=x[:,0],x[:,1],x[:,2]
            maximum=np.max(x[:,:3],axis=1);delta=maximum-np.min(x[:,:3],axis=1)
            saturation=np.divide(delta,maximum,out=np.zeros_like(delta),where=maximum>0)
            safe_delta=np.where(delta>0,delta,1)
            hue=np.where(r>=maximum,(g-b)/safe_delta,np.where(g>=maximum,2+(b-r)/safe_delta,4+(r-g)/safe_delta))/6
            hue=np.where(hue<0,hue+1,hue);hue=np.where(saturation>0,hue,0)
            v=np.column_stack((hue,saturation,maximum,x[:,3]))
        elif op==Op.HsvToRgb:
            x=registers[a];h=(x[:,0]-np.floor(x[:,0]))*6;s=x[:,1];brightness=x[:,2]
            sector=np.trunc(h).astype(int);fraction=h-sector
            p=brightness*(1-s);q=brightness*(1-s*fraction);t=brightness*(1-s*(1-fraction))
            choices=np.stack((np.column_stack((brightness,t,p)),np.column_stack((q,brightness,p)),
                              np.column_stack((p,brightness,t)),np.column_stack((p,q,brightness)),
                              np.column_stack((t,p,brightness)),np.column_stack((brightness,p,q))),axis=0)
            rgb=choices[sector,np.arange(count)]
            rgb=np.where((s<.0001)[:,None],brightness[:,None],rgb)
            v=np.column_stack((rgb,x[:,3]))
        elif op==Op.HexImage:
            # MaterialX v1.39.5 mx_hextile.glsl; explicit offline LOD zero.
            uv=registers[a][:,:2];primary,ranges,extra,luma=registers[c:c+4]
            skew=np.column_stack((uv[:,0]-.57735027*uv[:,1],1.15470054*uv[:,1]))*(np.sqrt(3)*2)
            f=skew-np.floor(skew);z=1-f[:,0]-f[:,1];s=(z<=0).astype(float);s2=2*s-1
            bary=np.column_stack((-z*s2,s-f[:,1]*s2,s-f[:,0]*s2))
            samples=[]
            for tile in range(3):
                shift=np.column_stack((s,s)) if tile==0 else (np.column_stack((s,1-s)) if tile==1 else np.column_stack((1-s,s)))
                ids=np.floor(skew)+shift
                center=np.column_stack((ids[:,0]+.5*ids[:,1],ids[:,1]/1.15470054))/(np.sqrt(3)*2)
                p=ids+.12345;p3=np.column_stack((p[:,0],p[:,1],p[:,0]))*np.array([.1031,.1030,.0973])
                p3-=np.floor(p3);p3+=np.sum(p3*(p3[:,[1,2,0]]+33.33),axis=1)[:,None]
                random=(p3[:,[0,0]]+p3[:,[1,2]])*p3[:,[2,1]];random-=np.floor(random)
                angle=(ranges[:,0]+(ranges[:,1]-ranges[:,0])*random[:,0]*primary[:,0])*np.pi/180
                scale=1+((ranges[:,2]+(ranges[:,3]-ranges[:,2])*random[:,1])-1)*primary[:,1]
                offset=extra[:,0:1]+(extra[:,1:2]-extra[:,0:1])*random*primary[:,2:3]
                p=uv-center;co=np.cos(angle);si=np.sin(angle)
                coord=np.column_stack((p[:,0]*co-p[:,1]*si,p[:,0]*si+p[:,1]*co))/scale[:,None]+center+offset
                samples.append(sample_texture(textures[b],coord,0))
            colors=np.stack(samples,axis=1)
            luminance=np.sum(colors[:,:,:3]*luma[:,None,:3],axis=2)
            luminance=1+(luminance-1)*extra[:,2:3]
            def weights(light):
                geometric=np.maximum(bary,0)**7;w=geometric*light
                w=np.where((np.sum(w,axis=1)<=0)[:,None],geometric,w)
                w/=np.maximum(np.sum(w,axis=1,keepdims=True),1e-30)
                gain=np.clip(primary[:,3:4],.001,.999);q=(1/gain-2)*(1-2*w)
                transformed=np.where(w<.5,w/(q+1),(q-w)/(q-1))
                w=np.where((primary[:,3:4]!=.5),transformed,w)
                return w/np.maximum(np.sum(w,axis=1,keepdims=True),1e-30)
            rgb=np.sum(colors[:,:,:3]*weights(luminance)[:,:,None],axis=1)
            alpha=np.sum(colors[:,:,3]*weights(np.ones_like(luminance)),axis=1)
            v=np.column_stack((rgb,alpha))
        elif op==Op.Normalmap: raise AssetError("Normalmap cannot be used as an offline displacement pattern")
        elif op==Op.Output: v=registers[a]
        else: raise AssetError("Unsupported CPU pattern opcode: "+str(op))
        if not np.isfinite(v).all(): raise AssetError("Nonfinite value evaluating MaterialX pattern")
        registers.append(v)
    return registers if supplied is not None else [v[0] for v in registers]
