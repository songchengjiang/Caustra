#!/usr/bin/env python3
"""Convert the pinned official PBRT San Miguel scene to editable USD/MaterialX.

Run with the asset-tools Python environment. Downloads are a separate, explicit
step (tools/fetch_sanmiguel.py); this program never accesses the network.
"""
import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade, Vt

from caustra_assets.model import AssetError
from caustra_assets.pbrt_scene import load_scene, read_ply


SOURCE_COMMIT="30cf4a0346ae5a80a2d7a530a3ef7d0fa4f70572"


def _matrix(value):
    return Gf.Matrix4d(*np.asarray(value,dtype=np.float64).T.ravel().tolist())


def _corners(bounds):
    low,high=bounds
    return np.array([[low[0] if i&1==0 else high[0],
                      low[1] if i&2==0 else high[1],
                      low[2] if i&4==0 else high[2],1.] for i in range(8)])


def _transformed_bounds(bounds,matrix):
    points=(_corners(bounds)@matrix.T)[:,:3]
    return points.min(axis=0),points.max(axis=0)


def _merge_bounds(bounds):
    bounds=list(bounds)
    if not bounds: raise AssetError("No geometric bounds in scene")
    return np.min([b[0] for b in bounds],axis=0),np.max([b[1] for b in bounds],axis=0)


def _shape_path(scene,shape):
    path=(scene.asset_root/str(shape.command.params["filename"].value)).resolve()
    if not path.is_relative_to(scene.asset_root): shape.command.fail("PLY filename escapes source directory")
    return path


def _deduplicate_vertices(data):
    """Merge only bit-identical position/normal/UV tuples, preserving seams."""
    points,indices,normals,uv=data
    attributes=[points]+([normals] if normals is not None else [])+([uv] if uv is not None else [])
    packed=np.ascontiguousarray(np.column_stack(attributes),dtype=np.float32)
    records=packed.view(np.dtype((np.void,packed.dtype.itemsize*packed.shape[1]))).reshape(-1)
    unused,first,inverse=np.unique(records,return_index=True,return_inverse=True)
    remapped=np.ascontiguousarray(inverse[indices],dtype=np.int32)
    return (np.ascontiguousarray(points[first]),remapped,
            np.ascontiguousarray(normals[first]) if normals is not None else None,
            np.ascontiguousarray(uv[first]) if uv is not None else None)


def _bump_metrics(scene,meshes):
    """Area-weighted authored dPdu/dPdv lengths, without baking large instances."""
    totals={};ranges={}
    for shape in scene.shapes+[s for shapes in scene.objects.values() for s in shapes]:
        points,indices,normals,uv=meshes[_shape_path(scene,shape)]
        if uv is None or not len(indices): continue
        # Work in the prototype's transformed world units. Repeated leaf/chair
        # instances retain their transform and do not multiply this statistic.
        p=points.astype(np.float64)@shape.transform[:3,:3].T
        e1=p[indices[:,1]]-p[indices[:,0]];e2=p[indices[:,2]]-p[indices[:,0]]
        t1=uv[indices[:,1]].astype(np.float64)-uv[indices[:,0]];t2=uv[indices[:,2]].astype(np.float64)-uv[indices[:,0]]
        det=t1[:,0]*t2[:,1]-t1[:,1]*t2[:,0];areas=np.linalg.norm(np.cross(e1,e2),axis=1)*.5
        valid=(np.abs(det)>1e-12)&(areas>1e-12)
        if not np.any(valid): continue
        e1=e1[valid];e2=e2[valid];t1=t1[valid];t2=t2[valid];det=det[valid];weights=areas[valid]
        lengths=np.column_stack((np.linalg.norm((e1*t2[:,1,None]-e2*t1[:,1,None])/det[:,None],axis=1),
                                 np.linalg.norm((e2*t1[:,0,None]-e1*t2[:,0,None])/det[:,None],axis=1)))
        if np.any(~np.isfinite(lengths)) or np.any(lengths<=0): continue
        name=shape.material;weighted=np.sum(lengths*weights[:,None],axis=0);weight=float(weights.sum())
        previous=totals.get(name,(np.zeros(2),0.));totals[name]=(previous[0]+weighted,previous[1]+weight)
        if name not in ranges: ranges[name]=(lengths.min(axis=0),lengths.max(axis=0))
        else: ranges[name]=(np.minimum(ranges[name][0],lengths.min(axis=0)),np.maximum(ranges[name][1],lengths.max(axis=0)))
    return {n:tuple((v/w).tolist()) for n,(v,w) in totals.items()}, {n:{"minimum":v[0].tolist(),"maximum":v[1].tolist()} for n,v in ranges.items()}


def _dielectric_topology(scene,meshes):
    """Detect open/nonmanifold solid dielectric geometry before GPU transport.

    PBRT tolerates these artist-authored window/water sheets. Caustra's real
    dielectric volume adapter requires closed, consistently oriented boundaries;
    the converter records an explicit thin-sheet approximation for those named
    source materials, rather than disabling runtime medium validation.
    """
    definitions=scene.material_definitions();records=[];overrides={};seen=set()
    for shape in scene.shapes+[s for group in scene.objects.values() for s in group]:
        if definitions[shape.material]["type"]!="dielectric": continue
        path=_shape_path(scene,shape);key=(path,shape.material)
        if key in seen: continue
        seen.add(key);points,indices,unused,uv=meshes[path]
        welded,inverse=np.unique(points,axis=0,return_inverse=True);triangles=inverse[indices]
        edges=np.concatenate((triangles[:,(0,1)],triangles[:,(1,2)],triangles[:,(2,0)]))
        signs=np.where(edges[:,0]<edges[:,1],1.,-1.)
        unused,edge_ids,counts=np.unique(np.sort(edges,axis=1),axis=0,return_inverse=True,return_counts=True)
        balance=np.bincount(edge_ids,weights=signs)
        boundary=int(np.count_nonzero(counts==1));nonmanifold=int(np.count_nonzero(counts>2))
        winding=int(np.count_nonzero((counts==2)&(balance!=0)))
        vertices=welded.astype(np.float64);vertices-=vertices.mean(axis=0)
        volume=float(np.einsum('ij,ij->i',vertices[triangles[:,0]],np.cross(vertices[triangles[:,1]],vertices[triangles[:,2]])).sum()/6)
        if shape.reverse: volume=-volume
        extent=float(np.max(np.ptp(vertices,axis=0)));positive=volume>max(1e-30,extent**3*1e-14)
        closed=boundary==0 and nonmanifold==0 and winding==0 and positive
        record={"file":str(path.relative_to(scene.asset_root)).replace("\\","/"),"material":shape.material,
                "position_welded_vertices":len(welded),"boundary_edges":boundary,"nonmanifold_edges":nonmanifold,
                "inconsistent_oriented_edges":winding,"signed_volume":volume,"closed_consistent_positive_volume":closed}
        records.append(record)
        if not closed:
            detail=f"{path.name}: {boundary} boundary edges, {nonmanifold} nonmanifold edges, {winding} inconsistent edges, volume={volume:.9g}"
            overrides[shape.material]=overrides.get(shape.material,"")+("; " if shape.material in overrides else "")+detail
    overrides={name:"Explicit thin-sheet approximation: source dielectric geometry does not form a valid closed oriented volume. "+reason for name,reason in overrides.items()}
    return overrides,records


def _strict_triangle_crossings(first,second,first_indices=None,second_indices=None,return_detail=False):
    """Find an interior edge/triangle crossing; AABB contact alone is ignored.

    Both edge directions are tested because the intersection segment can be
    bounded entirely by either triangle. Coplanar contacts and endpoint/edge
    coincidences do not establish interpenetrating volumes.
    """
    low=second.min(axis=1);high=second.max(axis=1)

    def edge_hits(triangles,other):
        edge1=other[:,1]-other[:,0];edge2=other[:,2]-other[:,0]
        for left,right in ((0,1),(1,2),(2,0)):
            origin=triangles[:,left];direction=triangles[:,right]-origin
            h=np.cross(direction,edge2);det=np.einsum("ij,ij->i",edge1,h)
            scale=np.linalg.norm(direction,axis=1)*np.linalg.norm(edge1,axis=1)*np.linalg.norm(edge2,axis=1)
            valid=np.abs(det)>np.maximum(1e-30,scale*1e-10)
            inv=np.divide(1.,det,out=np.zeros_like(det),where=valid)
            offset=origin-other[:,0];u=np.einsum("ij,ij->i",offset,h)*inv;q=np.cross(offset,edge1)
            v=np.einsum("ij,ij->i",direction,q)*inv;t=np.einsum("ij,ij->i",edge2,q)*inv
            valid&=(u>1e-7)&(v>1e-7)&(u+v<1-1e-7)&(t>1e-7)&(t<1-1e-7)
            hits=np.flatnonzero(valid)
            if len(hits):
                index=int(hits[0]);return (origin[index]+t[index]*direction[index]).tolist(),index,(left,right)
        return None

    for begin in range(0,len(first),128):
        batch=first[begin:begin+128];a_low=batch.min(axis=1);a_high=batch.max(axis=1)
        overlap=np.all((a_high[:,None,:]>=low[None,:,:])&(a_low[:,None,:]<=high[None,:,:]),axis=2)
        if first_indices is not None and second_indices is not None:
            # Source position-welded adjacency excludes shared edges/vertices
            # from self-intersection evidence, including attribute seams.
            adjacency=np.any(first_indices[begin:begin+128,None,:,None]==second_indices[None,:,None,:],axis=(2,3))
            overlap&=~adjacency
        left,right=np.nonzero(overlap)
        if not len(left): continue
        a=batch[left];b=second[right]
        hit=edge_hits(a,b);direction="first"
        if hit is None: hit=edge_hits(b,a);direction="second"
        if hit is not None:
            point,index,edge=hit
            if return_detail:
                return {"point":point,"first_triangle":int(begin+left[index]),"second_triangle":int(right[index]),
                        "edge_from":direction,"edge_vertices":list(edge)}
            return point
    return None


def _self_intersecting_dielectrics(scene,meshes,thin_overrides,topology):
    """Closed topology is necessary but does not establish a valid solid.

    Test every source mesh that passed the edge/volume audit for nonadjacent
    strict boundary crossings. Attribute-seam vertices are welded by position
    for adjacency only. A witnessed crossing is recorded and explicitly adapts
    that source material; unrelated valid glass remains solid.
    """
    records=[]
    for entry in topology:
        if not entry["closed_consistent_positive_volume"]:continue
        path=(scene.asset_root/entry["file"]).resolve();points,indices,unused,uv=meshes[path]
        unused,inverse=np.unique(points,axis=0,return_inverse=True);welded_indices=inverse[indices]
        triangles=points.astype(np.float64)[indices]
        witness=_strict_triangle_crossings(triangles,triangles,welded_indices,welded_indices,return_detail=True)
        row={"file":entry["file"],"material":entry["material"],"triangles":len(indices),
             "strict_nonadjacent_self_intersection":witness is not None,"witness":witness}
        if witness is not None:
            name=entry["material"]
            thin_overrides[name]="Explicit thin-sheet approximation: the closed source mesh contains a proven strict nonadjacent triangle self-intersection (%s, faces %d/%d). Closed edges and positive total volume do not define a valid solid in this case."%(entry["file"],witness["first_triangle"],witness["second_triangle"])
        records.append(row)
    return {"policy":"Position-welded nonadjacent strict triangle crossings; shared edges/vertices and AABB overlap alone do not trigger adaptation.","meshes":records}


def _overlapping_dielectrics(scene,meshes,thin_overrides):
    """Audit actual closed-solid instances, retaining isolated solid glass.

    An AABB sweep is only a broad phase. A strict interior triangle crossing
    proves the boundaries are not disjoint or nested; those individual
    placements receive a named thin-walled material derived from the source.
    """
    definitions=scene.material_definitions();candidates=[]

    def add(kind,index,shape,matrix,object_name=None):
        if definitions[shape.material]["type"]!="dielectric" or shape.material in thin_overrides:return
        points,indices,unused,uv=meshes[_shape_path(scene,shape)]
        world=points.astype(np.float64)@matrix[:3,:3].T+matrix[:3,3]
        candidates.append({"kind":kind,"index":index,"shape":shape,"matrix":matrix,
                           "object":object_name,"vertices":world,"indices":indices,
                           "minimum":world.min(axis=0),"maximum":world.max(axis=0)})
    for index,shape in enumerate(scene.shapes):add("direct_shape",index,shape,shape.transform)
    for index,instance in enumerate(scene.instances):
        group=scene.objects[instance.name]
        for shape in group:add("object_instance",index,shape,instance.transform@shape.transform,instance.name)
    candidates.sort(key=lambda c:float(c["minimum"][0]));active=[];broad_phase=0;conflicts=[];affected={}
    for current in candidates:
        active=[c for c in active if c["maximum"][0]>current["minimum"][0]]
        for other in active:
            if np.any(other["maximum"]<=current["minimum"]) or np.any(current["maximum"]<=other["minimum"]):continue
            broad_phase+=1
            duplicate=_shape_path(scene,current["shape"])==_shape_path(scene,other["shape"]) and np.array_equal(current["matrix"],other["matrix"])
            point=((current["minimum"]+current["maximum"])*.5).tolist() if duplicate else _strict_triangle_crossings(current["vertices"][current["indices"]],other["vertices"][other["indices"]])
            if point is None:continue
            def identity(item):return {"kind":item["kind"],"index":item["index"],"object":item["object"],"material":item["shape"].material,
                                      "source_ply":str(item["shape"].command.params["filename"].value)}
            conflicts.append({"first":identity(other),"second":identity(current),"crossing_world_z_up":point,"reason":"identical coincident placement" if duplicate else "strict interior edge/triangle crossing"})
            for item in (other,current):
                key=(item["kind"],item["index"])
                if key in affected and affected[key]!=item["shape"].material:
                    raise AssetError("Overlapping multi-material object requires per-mesh instance material variants")
                if item["kind"]=="object_instance" and len({s.material for s in scene.objects[item["object"]]})!=1:
                    raise AssetError("Overlapping multi-material object requires per-mesh instance material variants")
                affected[key]=item["shape"].material
        active.append(current)
    derived={}
    for original in sorted(set(affected.values())):
        name=original+"__interpenetrating_instance_thin"
        if name in scene.materials:raise AssetError("Derived overlap material name collides with source: "+name)
        scene.materials[name]=replace(scene.materials[original],args=(name,));derived[original]=name
        thin_overrides[name]="Explicit per-instance thin-sheet approximation: this placement has a proven non-nested intersection with another closed dielectric surface. Isolated placements retain the original solid material."
    mapping={key:derived[original] for key,original in affected.items()}
    return mapping,{"closed_solid_placements":len(candidates),"aabb_candidate_pairs":broad_phase,"confirmed_intersecting_pairs":len(conflicts),
                    "affected_placements":len(affected),"derived_materials":derived,"conflicts":conflicts,
                    "policy":"Only placements with strict triangle crossings or identical coincident geometry receive derived thin-walled materials; AABB overlap or valid nesting alone does not change a material."}


def _write_mesh(stage,path,shape,data,material):
    points,indices,normals,uv=data
    mesh=UsdGeom.Mesh.Define(stage,path)
    mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points))
    mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(indices),3,dtype=np.int32)))
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.ascontiguousarray(indices.reshape(-1))))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateOrientationAttr(UsdGeom.Tokens.leftHanded if shape.reverse else UsdGeom.Tokens.rightHanded)
    if normals is not None:
        mesh.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(-normals if shape.reverse else normals));mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    if uv is not None:
        UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st",Sdf.ValueTypeNames.TexCoord2fArray,UsdGeom.Tokens.vertex).Set(Vt.Vec2fArray.FromNumpy(uv))
    mesh.CreateExtentAttr([Gf.Vec3f(*points.min(axis=0).tolist()),Gf.Vec3f(*points.max(axis=0).tolist())])
    if not np.array_equal(shape.transform,np.eye(4)):
        UsdGeom.Xformable(mesh).AddTransformOp().Set(_matrix(shape.transform))
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    mesh.GetPrim().SetCustomDataByKey("pbrt:material",shape.material)
    mesh.GetPrim().SetCustomDataByKey("pbrt:source",str(shape.command.params["filename"].value).replace("\\","/"))
    return mesh


def _write_camera(stage,scene):
    if scene.camera is None: raise AssetError("Source has no camera")
    source=scene.camera.command
    if source.args!=("perspective",): source.fail("Only perspective camera is supported")
    allowed={"fov","focaldistance","lensradius","shutteropen","shutterclose"}
    if set(source.params)-allowed: source.fail("Unsupported camera parameters: "+str(set(source.params)-allowed))
    value=lambda name,default:float(source.params[name].value) if name in source.params else default
    if value("lensradius",0)!=0: source.fail("Depth-of-field conversion is unsupported")
    if scene.film is None or scene.film.args!=("rgb",): source.fail("Expected an rgb film")
    film=scene.film.params;w=int(film["xresolution"].value);h=int(film["yresolution"].value)
    aspect=w/h;fov=value("fov",90)
    if not 0<fov<180 or w<=0 or h<=0: source.fail("Invalid perspective film/fov")
    # PBRT defines fov across the shorter dimension. USD stores a physical
    # filmback; its default camera looks down -Z instead of PBRT's +Z.
    vertical=20.955/max(aspect,1.);horizontal=vertical*aspect
    focal=min(horizontal,vertical)/(2*math.tan(math.radians(fov)*.5))
    camera=UsdGeom.Camera.Define(stage,"/World/Camera")
    camera.CreateProjectionAttr(UsdGeom.Tokens.perspective)
    camera.CreateHorizontalApertureAttr(horizontal);camera.CreateVerticalApertureAttr(vertical);camera.CreateFocalLengthAttr(focal)
    camera.CreateClippingRangeAttr(Gf.Vec2f(.001,1e8))
    # No authored focus distance means the runtime chooses an orbit target
    # inside scene bounds, rather than PBRT's effectively infinite DOF focus.
    camera.CreateFocusDistanceAttr(0)
    camera.AddTransformOp().Set(_matrix(scene.camera.world_from_camera@np.diag((1.,1.,-1.,1.))))
    crop=list(film["cropwindow"].values) if "cropwindow" in film else [0,1,0,1]
    metadata={"resolution":[w,h],"crop_window":crop,"iso":float(film["iso"].value) if "iso" in film else 100.,
              "fov_short_axis_degrees":fov,"fov_vertical_degrees":math.degrees(2*math.atan(vertical/(2*focal))),
              "source_focus_distance":value("focaldistance",1e6),"focus_distance_policy":"runtime orbit target from geometry bounds",
              "crop_policy":"full film is displayed; original crop metadata retained (no asymmetric frustum in runtime)",
              "iso_policy":"source ISO recorded; display exposure is controlled by Caustra"}
    camera.GetPrim().SetCustomDataByKey("pbrt:filmMetadata",json.dumps(metadata,sort_keys=True))
    return metadata


def convert(source,output,environment_callback=None):
    start=time.monotonic();source=Path(source).resolve();output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    print("Reading PBRT scene...",flush=True);scene=load_scene(source)
    shapes=scene.shapes+[s for group in scene.objects.values() for s in group]
    paths=sorted({_shape_path(scene,s) for s in shapes});meshes={};degenerate_meshes=[];source_triangles=0;source_vertices=0
    for i,path in enumerate(paths):
        data=read_ply(path);points,indices,normals,uv=data;source_triangles+=len(indices);source_vertices+=len(points)
        edges1=points[indices[:,1]].astype(np.float64)-points[indices[:,0]]
        edges2=points[indices[:,2]].astype(np.float64)-points[indices[:,0]]
        valid=np.any(np.cross(edges1,edges2)!=0,axis=1)
        if not np.all(valid):
            degenerate_meshes.append({"file":str(path.relative_to(scene.asset_root)).replace("\\","/"),"triangles":int(np.count_nonzero(~valid))})
            indices=np.ascontiguousarray(indices[valid]);data=(points,indices,normals,uv)
        if not len(indices): raise AssetError("PLY has no nondegenerate triangles: "+str(path))
        meshes[path]=_deduplicate_vertices(data)
        if (i+1)%100==0: print(f"Read PLY {i+1}/{len(paths)}",flush=True)
    metrics,metric_ranges=_bump_metrics(scene,meshes)
    thin_overrides,dielectric_audit=_dielectric_topology(scene,meshes)
    self_intersection_audit=_self_intersecting_dielectrics(scene,meshes,thin_overrides,dielectric_audit)
    placement_materials,overlap_audit=_overlapping_dielectrics(scene,meshes,thin_overrides)
    for original,derived in overlap_audit["derived_materials"].items():
        if original in metrics:metrics[derived]=metrics[original]
    stage=Usd.Stage.CreateNew(str(output/"scene.usdc"))
    UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z);UsdGeom.SetStageMetersPerUnit(stage,1.)
    world=UsdGeom.Xform.Define(stage,"/World");stage.SetDefaultPrim(world.GetPrim())
    from caustra_assets.pbrt_materials import MaterialLibrary
    print("Converting material graphs and textures...",flush=True)
    library=MaterialLibrary(scene.material_definitions(),scene.texture_definitions(),scene.asset_root,output,
                            bump_metrics=metrics,thin_walled_overrides=thin_overrides)
    materials=library.write(stage)
    camera_report=_write_camera(stage,scene)
    stage.CreateClassPrim("/__Prototypes")
    UsdGeom.Scope.Define(stage,"/World/Geometry");UsdGeom.Scope.Define(stage,"/World/Instances")
    object_paths={};object_bounds={};object_triangles={};bounds=[]
    for number,(name,group) in enumerate(scene.objects.items()):
        path=f"/__Prototypes/Object_{number:04d}";object_paths[name]=path
        UsdGeom.Xform.Define(stage,path);group_bounds=[];triangle_count=0
        for i,shape in enumerate(group):
            data=meshes[_shape_path(scene,shape)]
            _write_mesh(stage,path+f"/Mesh_{i:04d}",shape,data,materials[shape.material])
            group_bounds.append(_transformed_bounds((data[0].min(axis=0),data[0].max(axis=0)),shape.transform));triangle_count+=len(data[1])
        if not group_bounds: raise AssetError("Empty Object definition "+name)
        object_bounds[name]=_merge_bounds(group_bounds);object_triangles[name]=triangle_count
    for i,shape in enumerate(scene.shapes):
        data=meshes[_shape_path(scene,shape)]
        material_name=placement_materials.get(("direct_shape",i),shape.material)
        _write_mesh(stage,f"/World/Geometry/Mesh_{i:04d}",shape,data,materials[material_name])
        bounds.append(_transformed_bounds((data[0].min(axis=0),data[0].max(axis=0)),shape.transform))
    # A live UsdStage recomposes after each edit. At 160k leaf instances this
    # grows needlessly expensive; author equivalent Sdf specs in one change
    # block, then let USD compose the complete native instance hierarchy once.
    layer=stage.GetRootLayer()
    with Sdf.ChangeBlock():
        for i,instance in enumerate(scene.instances):
            prim=Sdf.CreatePrimInLayer(layer,f"/World/Instances/Instance_{i:06d}")
            prim.specifier=Sdf.SpecifierDef;prim.typeName="Xform";prim.instanceable=True
            prim.referenceList.prependedItems=[Sdf.Reference("",Sdf.Path(object_paths[instance.name]))]
            attribute=Sdf.AttributeSpec(prim,"xformOp:transform",Sdf.ValueTypeNames.Matrix4d,Sdf.VariabilityVarying)
            attribute.default=_matrix(instance.transform)
            order=Sdf.AttributeSpec(prim,"xformOpOrder",Sdf.ValueTypeNames.TokenArray,Sdf.VariabilityUniform)
            order.default=Vt.TokenArray(["xformOp:transform"])
            override=placement_materials.get(("object_instance",i))
            if override:
                api=Sdf.TokenListOp();api.prependedItems=["MaterialBindingAPI"];prim.SetInfo("apiSchemas",api)
                relationship=Sdf.RelationshipSpec(prim,"material:binding",False)
                relationship.targetPathList.explicitItems=[materials[override].GetPath()]
                relationship.SetInfo("bindMaterialAs",UsdShade.Tokens.strongerThanDescendants)
            bounds.append(_transformed_bounds(object_bounds[instance.name],instance.transform))
            if (i+1)%20000==0: print(f"Authored USD instances {i+1}/{len(scene.instances)}",flush=True)
    geometry_bounds=_merge_bounds(bounds)
    world.GetPrim().SetCustomDataByKey("caustra:sourceGeometryBounds",Vt.DoubleArray([float(v) for b in geometry_bounds for v in b]))
    if environment_callback is None:
        from caustra_assets.pbrt_environment import add_environment
        environment_callback=add_environment
    print("Converting HDR environment...",flush=True)
    environment_report=environment_callback(stage,scene,library,output,geometry_bounds)
    library.write_mtlx(output/"materials.mtlx")
    report={"source_repository":"https://github.com/mmp/pbrt-v4-scenes","source_commit":SOURCE_COMMIT,
            "source_scene":source.name,"source_pbrt_files":len(scene.source_files),
            "direct_shapes":len(scene.shapes),"object_definitions":len(scene.objects),"object_instances":len(scene.instances),
            "unique_ply_files":len(meshes),"unique_vertices":sum(len(d[0]) for d in meshes.values()),
            "source_vertices":source_vertices,"vertex_deduplication":"bit-identical full position/normal/UV tuples; authored attribute seams retained",
            "unique_triangles":sum(len(d[1]) for d in meshes.values()),
            "source_triangles":source_triangles,"discarded_zero_area_triangles":sum(d["triangles"] for d in degenerate_meshes),
            "zero_area_triangles_by_mesh":degenerate_meshes,
            "instanced_triangles":sum(len(meshes[_shape_path(scene,s)][1]) for s in scene.shapes)+sum(object_triangles[i.name] for i in scene.instances),
            "material_definitions":len(scene.materials),"texture_definitions":len(scene.textures),
            "renamed_source_definitions":scene.renamed_definitions,"command_counts":scene.command_counts,
            "geometry_bounds_z_up":[b.tolist() for b in geometry_bounds],"camera":camera_report,
            "materials":library.report,"bump_metric_ranges":metric_ranges,
            "dielectric_topology":dielectric_audit,"thin_walled_material_overrides":thin_overrides,
            "dielectric_placement_overlap":overlap_audit,
            "dielectric_self_intersection":self_intersection_audit,
            "bump_metric_policy":"area-weighted prototype dPdu/dPdv lengths; instance scaling is not baked into normal maps",
            "environment":environment_report,
            "render_options":[{"command":c.name,"implementation":list(c.args),"parameters":{n:p.value for n,p in c.params.items()}} for c in scene.options],
            "source_files":[{"path":str(p.relative_to(scene.asset_root)).replace("\\","/"),"sha256":hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(scene.source_files)],
            "elapsed_seconds":round(time.monotonic()-start,3)}
    stage.GetRootLayer().customLayerData={"pbrt:sourceCommit":SOURCE_COMMIT,"pbrt:sourceScene":source.name}
    stage.GetRootLayer().Save()
    (output/"conversion_report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(f"Wrote {output/'scene.usdc'} ({len(scene.instances)} native instances, {report['instanced_triangles']} rendered triangles)",flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",type=Path,default=Path("build/deps/pbrt-v4-scenes/sanmiguel/sanmiguel-courtyard-second.pbrt"))
    parser.add_argument("--output",type=Path,default=Path("renders/sanmiguel"))
    args=parser.parse_args()
    try: convert(args.source,args.output)
    except (AssetError,OSError,ValueError) as error: parser.exit(1,f"San Miguel conversion: {error}\n")


if __name__=="__main__": main()
