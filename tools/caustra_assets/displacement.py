"""Bounded offline microtriangle displacement; no runtime importer dependency."""
import numpy as np
from .model import AssetError, Mesh
from .pattern_cpu import evaluate_program, normalize

MAX_TRIANGLES=1_000_000


def displace_mesh(mesh,displacement,textures,transform,subdivisions):
    program,result,kind=displacement
    factor=1<<subdivisions
    if len(mesh.indices)//3*factor*factor>MAX_TRIANGLES:
        raise AssetError("Displacement exceeds the one-million-triangle mesh limit; reduce --displacement-subdivisions")
    source=np.asarray(mesh.vertices,dtype=float);triangles=np.asarray(mesh.indices).reshape(-1,3)
    # Missing normals acquire a smooth area-weighted basis at shared positions.
    keys=[tuple(np.round(v[:3],10)) for v in source];basis={}
    for indices in triangles:
        p,q,r=source[indices,:3];n=np.cross(q-p,r-p)
        for index in indices: basis[keys[index]]=basis.get(keys[index],np.zeros(3))+n
    for i,v in enumerate(source):
        if np.linalg.norm(v[4:7])<1e-12: source[i,4:7]=normalize(basis[keys[i]])
    tangent_basis={};vector_basis=[]
    tangent_keys=[tuple(np.round(np.r_[v[:3],v[4:7],v[8:10]],9)) for v in source]
    for tri in triangles:
        p,q,r=source[tri,:3];uv0,uv1,uv2=source[tri,8:10]
        u,v=uv1-uv0,uv2-uv0;det=u[0]*v[1]-u[1]*v[0]
        if kind=="vector3":
            if abs(det)<=max(np.linalg.norm(u)*np.linalg.norm(v),1e-30)*1e-10:
                raise AssetError("Vector displacement requires nondegenerate UV derivatives (dPdu, dPdv, N)")
            vector_basis.append((((q-p)*v[1]-(r-p)*u[1])/det,
                                 ((r-p)*u[0]-(q-p)*v[0])/det))
        t=((q-p)*v[1]-(r-p)*u[1])/det if abs(det)>1e-12 else np.zeros(3)
        for index in tri: tangent_basis[tangent_keys[index]]=tangent_basis.get(tangent_keys[index],np.zeros(3))+t
    for i,v in enumerate(source):
        n=normalize(v[4:7]);t=tangent_basis[tangent_keys[i]];t-=n*np.dot(n,t)
        if np.linalg.norm(t)<1e-12: t=np.cross([0,1,0] if abs(n[1])<.99 else [1,0,0],n)
        source[i,12:15]=normalize(t)
    bary=[];lookup={}
    for i in range(factor+1):
        for j in range(factor+1-i):
            lookup[i,j]=len(bary);bary.append((1-(i+j)/factor,i/factor,j/factor))
    local=[]
    for i in range(factor):
        for j in range(factor-i):
            local.extend((lookup[i,j],lookup[i+1,j],lookup[i,j+1]))
            if i+j<factor-1: local.extend((lookup[i+1,j],lookup[i+1,j+1],lookup[i,j+1]))
    bary=np.asarray(bary);local=np.asarray(local,dtype=np.int64);count=len(bary)
    vertices=np.concatenate([bary@source[triangle] for triangle in triangles])
    indices=np.concatenate([local+i*count for i in range(len(triangles))])
    positions=vertices[:,:3].copy();normals=normalize(vertices[:,4:7])
    matrix=np.asarray(transform,dtype=float).reshape(3,4)
    if abs(np.linalg.det(matrix[:,:3]))<1e-20: raise AssetError("Displacement instance transform is singular")
    world=positions@matrix[:,:3].T+matrix[:,3]
    world_normal=normalize(normals@np.linalg.inv(matrix[:,:3]))
    world_tangent=vertices[:,12:15]@matrix[:,:3].T
    world_tangent=normalize(world_tangent-world_normal*np.sum(world_tangent*world_normal,axis=-1,keepdims=True))
    values=evaluate_program(program,textures,uv=vertices[:,8:10],position=world,
                            object_position=positions,normal=world_normal,tangent=world_tangent)[result]
    if kind=="float": offset=normals*values[:,:1]
    else:
        # MaterialX ND_displacement_vector3 specifies (dPdu,dPdv,N),
        # not object XYZ: derivative magnitude and UV handedness matter.
        dpdu=np.repeat(np.asarray([basis[0] for basis in vector_basis]),count,axis=0)
        dpdv=np.repeat(np.asarray([basis[1] for basis in vector_basis]),count,axis=0)
        offset=dpdu*values[:,0:1]+dpdv*values[:,1:2]+normals*values[:,2:3]
    vertices[:,:3]+=offset
    if not np.isfinite(vertices).all(): raise AssetError("Displacement produced nonfinite geometry")
    # Keep UV seams as separate vertices; demand the same geometric result on
    # both sides. Inconsistent seam heights cannot form a sound BLAS boundary.
    extent=max(1.,np.max(np.ptp(positions,axis=0)));seams={}
    for before,after in zip(positions,vertices[:,:3]):
        key=tuple(np.round(before,9))
        if key in seams and np.linalg.norm(seams[key]-after)>extent*1e-6:
            raise AssetError("Displacement is discontinuous at a UV/normal seam; use a continuous height/vector field")
        seams[key]=after
    # Rebuild normals from displaced microgeometry, sharing across UV seams
    # only when the original authored smoothing normals agree.
    smooth=[tuple(np.round(np.r_[p,n],9)) for p,n in zip(positions,normals)];accum={}
    for tri in indices.reshape(-1,3):
        p,q,r=vertices[tri,:3];normal=np.cross(q-p,r-p)
        if np.linalg.norm(normal)<1e-18: raise AssetError("Displacement produced degenerate triangles")
        for index in tri: accum[smooth[index]]=accum.get(smooth[index],np.zeros(3))+normal
    vertices[:,4:7]=normalize(np.asarray([accum[key] for key in smooth]));vertices[:,12:16]=0
    return Mesh(mesh.material,vertices.tolist(),indices.tolist())
