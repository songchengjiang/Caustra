"""USD scene adapter kept outside the rendering core."""
from collections import defaultdict
from pathlib import Path
import math
import MaterialX as mx
from pxr import Ar, Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade
from .model import Asset, AssetError, Material, Mesh
from .materialx_compiler import MaterialCompiler, load_materialx, read_resource


def _value_string(value):
    if isinstance(value,Sdf.AssetPath): return value.resolvedPath or value.path
    if isinstance(value,bool): return "true" if value else "false"
    if isinstance(value,(float,int,str)): return str(value)
    try: return ", ".join(str(v) for v in value)
    except TypeError as error: raise AssetError("Unsupported USD material value: "+str(type(value))) from error


def materialx_from_usd(material,libraries,time=Usd.TimeCode.Default()):
    """Reconstruct the actual MaterialX node graph, without a PBR-value fallback."""
    output=material.GetSurfaceOutput("mtlx")
    if output and not output.HasConnectedSource(): output=None
    doc=mx.createDocument();doc.setDataLibrary(libraries)
    translated={};visiting=set();loaded_definition_assets=set()

    def resolve(port,seen=None):
        seen=set() if seen is None else set(seen)
        key=str(port.GetAttr().GetPath())
        if key in seen: raise AssetError("Cycle in USD material interface: "+key)
        seen.add(key)
        sources,invalid=port.GetConnectedSources()
        if invalid or len(sources)>1: raise AssetError("Ambiguous/broken USD material connection: "+str(port.GetAttr().GetPath()))
        if not sources: return None,port.Get(time)
        source=sources[0]
        api=source.source;name=str(source.sourceName)
        if source.sourceType==UsdShade.AttributeType.Input:
            return resolve(api.GetInput(name),seen)
        node=UsdShade.Shader(api.GetPrim())
        if node: return (convert(node),name),None
        output_port=api.GetOutput(name)
        if not output_port: raise AssetError("Unresolved USD nodegraph output")
        return resolve(output_port,seen)

    def convert(shader):
        path=str(shader.GetPath())
        if path in visiting: raise AssetError("Cycle in USD material graph: "+path)
        if path in translated: return translated[path]
        identifier=str(shader.GetIdAttr().Get() or "")
        if not identifier: identifier=str(shader.GetSourceAssetSubIdentifier("mtlx") or "")
        definition=libraries.getNodeDef(identifier)
        if definition is None:
            # MaterialX permits documents to define local node implementations
            # (the official SimpleHair example does this). USD's sourceAsset
            # metadata carries those definitions; keep the real graph rather
            # than guessing values for an otherwise unknown shader identifier.
            source=shader.GetSourceAsset("mtlx")
            if source and source.path:
                source_path=source.resolvedPath or source.path
                if not source.resolvedPath:
                    attr=shader.GetPrim().GetAttribute("info:mtlx:sourceAsset")
                    stack=attr.GetPropertyStack() if attr else []
                    if stack: source_path=Sdf.ComputeAssetPathRelativeToLayer(stack[0].layer,source.path)
                if source_path not in loaded_definition_assets:
                    local=mx.createDocument()
                    try:
                        mx.readFromXmlString(local,read_resource(source_path).decode("utf-8"),mx.FileSearchPath(str(Path(source_path).parent)))
                    except (OSError,ValueError,RuntimeError) as error:
                        raise AssetError("Cannot load local MaterialX definitions %s: %s" % (source_path,error)) from error
                    if local.getNodeDef(identifier) is None:
                        raise AssetError("Local MaterialX source has no NodeDef "+identifier)
                    libraries.importLibrary(local)
                    loaded_definition_assets.add(source_path)
                definition=libraries.getNodeDef(identifier)
        if definition is None: raise AssetError("Unknown MaterialX NodeDef %s on %s" % (identifier,path))
        visiting.add(path)
        name=mx.createValidName(path.strip("/").replace("/","_"))
        # Preserve distinct USD names even after MaterialX identifier sanitization.
        name=doc.createValidChildName(name)
        node=doc.addNodeInstance(definition,name);translated[path]=node
        for usd_input in shader.GetInputs():
            name=str(usd_input.GetBaseName());expected=definition.getActiveInput(name)
            if expected is None: raise AssetError("Unknown MaterialX input %s.%s" % (identifier,name))
            connection,value=resolve(usd_input)
            if connection is None and value is None: continue
            port=node.addInput(name,expected.getType())
            if connection:
                upstream,source_name=connection
                port.setConnectedNode(upstream)
                if upstream.getType()=="multioutput": port.setOutputString(source_name)
                elif source_name not in ("out",""):
                    outputs=upstream.getNodeDef().getActiveOutputs()
                    if len(outputs)>1: port.setOutputString(source_name)
            else:
                if isinstance(value,Sdf.AssetPath) and not value.resolvedPath:
                    stack=usd_input.GetAttr().GetPropertyStack()
                    if stack: value=Sdf.AssetPath(Sdf.ComputeAssetPathRelativeToLayer(stack[0].layer,value.path))
                port.setValueString(_value_string(value))
            space=usd_input.GetAttr().GetMetadata("colorSpace")
            if space: port.setColorSpace(str(space))
        visiting.remove(path)
        return node

    surface=resolve(output)[0][0] if output else None
    material_node=None
    if surface is not None:
        material_node=doc.addNode("surfacematerial","asset_material","material")
        material_node.addInput("surfaceshader","surfaceshader").setConnectedNode(surface)
    for terminal_name,accessor,kind in (("displacement",material.GetDisplacementOutput,"displacementshader"),("volume",material.GetVolumeOutput,"volumeshader")):
        port=next((p for p in (accessor("mtlx"),accessor("")) if p and p.HasConnectedSource()),None)
        if port is None: continue
        connection,_=resolve(port)
        if not connection: raise AssetError("Unresolved USD "+terminal_name+" terminal")
        node,_=connection
        if node.getType()!=kind: raise AssetError("Wrong MaterialX "+terminal_name+" terminal type")
        if kind=="volumeshader":
            wrapper=doc.addNodeInstance(libraries.getNodeDef("ND_volumematerial"),"asset_volume_material")
        else:
            if material_node is None: material_node=doc.addNode("surfacematerial","asset_material","material")
            wrapper=material_node
        wrapper.addInput(kind,kind).setConnectedNode(node)
    if surface is None and not any(n.getType()=="volumeshader" for n in doc.getNodes()):
        raise AssetError("Material %s has no MaterialX surface or volume; export with MaterialX enabled" % material.GetPath())
    valid,message=doc.validate()
    if not valid: raise AssetError("USD MaterialX graph is invalid: "+message)
    return doc,surface


def triangulate(points):
    """Ear clip planar polygons, preserving face-corner attributes and winding."""
    if len(points)<3: raise AssetError("USD face has fewer than three vertices")
    normal=[0.,0.,0.]
    for p,q in zip(points,points[1:]+points[:1]):
        normal[0]+=(p[1]-q[1])*(p[2]+q[2]);normal[1]+=(p[2]-q[2])*(p[0]+q[0]);normal[2]+=(p[0]-q[0])*(p[1]+q[1])
    drop=max(range(3),key=lambda i:abs(normal[i]));axes=[i for i in range(3) if i!=drop]
    xy=[(float(p[axes[0]]),float(p[axes[1]])) for p in points]
    cross=lambda a,b,c:(b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    area=sum(a[0]*b[1]-b[0]*a[1] for a,b in zip(xy,xy[1:]+xy[:1]))
    extent=max(max(p[i] for p in xy)-min(p[i] for p in xy) for i in range(2))
    epsilon=max(1e-24,extent*extent*1e-12)
    if abs(area)<=epsilon: raise AssetError("Degenerate USD polygon")
    sign=1 if area>0 else -1;remaining=list(range(len(points)));triangles=[]
    while len(remaining)>3:
        for i,b in enumerate(remaining):
            a=remaining[i-1];c=remaining[(i+1)%len(remaining)]
            if sign*cross(xy[a],xy[b],xy[c])<=epsilon: continue
            inside=lambda p: all(sign*cross(x,y,p)>=-epsilon for x,y in ((xy[a],xy[b]),(xy[b],xy[c]),(xy[c],xy[a])))
            if any(inside(xy[j]) for j in remaining if j not in (a,b,c)): continue
            triangles.append((a,b,c));remaining.pop(i);break
        else: raise AssetError("Cannot triangulate non-simple or degenerate USD polygon")
    triangles.append(tuple(remaining))
    return triangles


def import_usd(path,material_override=None,time=None,displacement_subdivisions=3):
    path=Path(path).resolve()
    try: stage=Usd.Stage.Open(str(path))
    except Exception as error: raise AssetError("Cannot open USD scene: "+str(error)) from error
    if not stage: raise AssetError("Cannot open USD scene: "+str(path))
    asset=Asset();compiler=MaterialCompiler(asset)
    snapshot=Usd.TimeCode(time) if time is not None else Usd.TimeCode.Default()
    cache=UsdGeom.XformCache(snapshot)
    meters=float(UsdGeom.GetStageMetersPerUnit(stage));up=str(UsdGeom.GetStageUpAxis(stage))
    if not math.isfinite(meters) or meters<=0: raise AssetError("Invalid USD metersPerUnit")
    compiler.unit_scale=meters
    if not isinstance(displacement_subdivisions,int) or not 0<=displacement_subdivisions<=7: raise AssetError("Displacement subdivisions must be between 0 and 7")
    if up not in ("Y","Z"): raise AssetError("USD upAxis must be Y or Z")
    def axis(p): return [float(p[0])*meters,float(p[1 if up=="Y" else 2])*meters,float(p[2])*meters if up=="Y" else -float(p[1])*meters]
    def transform(matrix):
        columns=[axis([matrix[c][0],matrix[c][1],matrix[c][2]]) for c in range(4)]
        return [columns[c][r] for r in range(3) for c in range(4)]
    def world_point(matrix,point): return axis(matrix.Transform(Gf.Vec3d(*point)))
    def attr_value(values,interpolation,face,corner,vertex,name):
        if not values: return None
        indices={"constant":0,"uniform":face,"vertex":vertex,"varying":vertex,"faceVarying":corner}
        if interpolation not in indices: raise AssetError("Unsupported USD %s interpolation: %s" % (name,interpolation))
        index=indices[interpolation]
        if index>=len(values): raise AssetError("USD %s attribute count does not match interpolation" % name)
        return values[index]
    materials={};interior_uv_mappings={}
    override=None
    if material_override:
        document,surface=load_materialx(material_override,compiler.libraries)
        override=compiler.compile(document,surface,Path(material_override).resolve().parent)
    def material_index(prim):
        if override is not None: return override
        binding=UsdShade.MaterialBindingAPI(prim)
        material=binding.ComputeBoundMaterial()[0]
        key=str(material.GetPath()) if material else "<default>"
        if key not in materials:
            if material: document,surface=materialx_from_usd(material,compiler.libraries,snapshot)
            else:
                document=mx.createDocument();document.setDataLibrary(compiler.libraries)
                surface=document.addNodeInstance(compiler.libraries.getNodeDef("ND_standard_surface_surfaceshader"),"default_surface")
            materials[key]=compiler.compile(document,surface,path.parent)
        return materials[key]
    bounds=[];camera_has_focus=False
    def area_light(prim):
        if prim.IsA(UsdLux.RectLight):
            light=UsdLux.RectLight(prim)
            width=float(light.GetWidthAttr().Get(snapshot));height=float(light.GetHeightAttr().Get(snapshot))
            if width<=0 or height<=0: raise AssetError("USD rectangle light dimensions must be positive")
            points=[(-width/2,-height/2,0),(-width/2,height/2,0),(width/2,height/2,0),(width/2,-height/2,0)]
            indices=[0,1,2,0,2,3]
            texture=light.GetTextureFileAttr().Get(snapshot)
            if texture and texture.path: raise AssetError("Textured USD area lights are unsupported")
        elif prim.IsA(UsdLux.DiskLight):
            light=UsdLux.DiskLight(prim);radius=float(light.GetRadiusAttr().Get(snapshot))
            if radius<=0: raise AssetError("USD disk light radius must be positive")
            segments=96
            points=[(0,0,0)]+[(radius*math.cos(-2*math.pi*i/segments),radius*math.sin(-2*math.pi*i/segments),0) for i in range(segments)]
            indices=[index for i in range(segments) for index in (0,1+i,1+(i+1)%segments)]
        else: raise AssetError("Unsupported USD light type: "+str(prim.GetTypeName()))
        if prim.HasAPI(UsdLux.ShapingAPI): raise AssetError("Shaped USD area lights are unsupported")
        intensity=float(light.GetIntensityAttr().Get(snapshot));exposure=float(light.GetExposureAttr().Get(snapshot))
        color=[float(v) for v in light.GetColorAttr().Get(snapshot)]
        if not math.isfinite(intensity) or intensity<0 or not math.isfinite(exposure) or any(not math.isfinite(v) or v<0 for v in color):
            raise AssetError("Invalid USD area light radiometry")
        if light.GetEnableColorTemperatureAttr().Get(snapshot):
            temperature=float(light.GetColorTemperatureAttr().Get(snapshot))
            if not math.isfinite(temperature) or temperature<=0: raise AssetError("Invalid USD light color temperature")
            if not hasattr(UsdLux,"BlackbodyTemperatureAsRgb"): raise AssetError("This USD build cannot evaluate color temperature")
            color=[a*float(b) for a,b in zip(color,UsdLux.BlackbodyTemperatureAsRgb(temperature))]
        matrix=cache.GetLocalToWorldTransform(prim)
        world=[world_point(matrix,p) for p in points]
        area=0.
        for i in range(0,len(indices),3):
            p,q,r=[world[index] for index in indices[i:i+3]]
            a=[q[j]-p[j] for j in range(3)];b=[r[j]-p[j] for j in range(3)]
            cross=[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
            area+=math.sqrt(sum(v*v for v in cross))*.5
        if area<=1e-20: raise AssetError("Degenerate USD area light transform")
        try: radiance=intensity*math.exp2(exposure)
        except OverflowError as error: raise AssetError("USD light exposure overflows linear radiance") from error
        if light.GetNormalizeAttr().Get(snapshot): radiance/=area
        material=Material();material.values[0]=[0,0,0,1];material.values[1]=[radiance*v for v in color]+[0]
        material.values[2][2]=1 # Preserve USD area lights' front-side-only emission.
        material.values[3][0]=0
        if any(not math.isfinite(v) for v in material.values[1]): raise AssetError("USD light radiance is not finite")
        mesh=Mesh(len(asset.materials),[list(p)+[0]+[0,0,-1,0]+[0,0,0,0]+[0,0,0,0] for p in points],indices)
        asset.materials.append(material);asset.instances.append((len(asset.meshes),transform(matrix)));asset.meshes.append(mesh)
        bounds.extend(world)
    with Ar.ResolverContextBinder(stage.GetPathResolverContext()):
        for prim in Usd.PrimRange(stage.GetPseudoRoot(),Usd.TraverseInstanceProxies()):
            if not prim.IsActive() or not prim.IsDefined(): continue
            imageable=UsdGeom.Imageable(prim)
            if imageable and imageable.ComputeVisibility(snapshot)==UsdGeom.Tokens.invisible: continue
            if prim.HasAPI(UsdLux.LightAPI): area_light(prim);continue
            if prim.IsA(UsdGeom.Camera) and not asset.has_camera:
                camera=UsdGeom.Camera(prim)
                if camera.GetProjectionAttr().Get(snapshot)!=UsdGeom.Tokens.perspective: raise AssetError("Orthographic USD cameras are unsupported")
                matrix=cache.GetLocalToWorldTransform(prim)
                focus=float(camera.GetFocusDistanceAttr().Get(snapshot) or 0)
                camera_has_focus=math.isfinite(focus) and focus>0
                position=world_point(matrix,(0,0,0));target=world_point(matrix,(0,0,-focus if camera_has_focus else -1));up_point=world_point(matrix,(0,1,0))
                direction=[up_point[i]-position[i] for i in range(3)]
                focal=float(camera.GetFocalLengthAttr().Get(snapshot));aperture=float(camera.GetVerticalApertureAttr().Get(snapshot))
                if focal<=0 or aperture<=0: raise AssetError("Invalid USD camera lens")
                asset.camera=position+[0]+target+[0]+direction+[0]+[math.degrees(2*math.atan(aperture/(2*focal)))]
                asset.has_camera=True
            if not prim.IsA(UsdGeom.Mesh): continue
            mesh=UsdGeom.Mesh(prim)
            if mesh.GetSubdivisionSchemeAttr().Get(snapshot)!=UsdGeom.Tokens.none:
                raise AssetError("Subdivision USD meshes require tessellation before export: "+str(prim.GetPath()))
            points=mesh.GetPointsAttr().Get(snapshot)
            counts=mesh.GetFaceVertexCountsAttr().Get(snapshot);indices=mesh.GetFaceVertexIndicesAttr().Get(snapshot)
            if not points or not counts: continue
            if not indices or sum(counts)!=len(indices) or any(i<0 or i>=len(points) for i in indices): raise AssetError("Invalid USD mesh topology")
            normals=mesh.GetNormalsAttr().Get(snapshot);normal_interpolation=str(mesh.GetNormalsInterpolation())
            default_material=material_index(prim)
            face_material=[default_material]*len(counts);assigned=set()
            for subset in UsdShade.MaterialBindingAPI(prim).GetMaterialBindSubsets():
                index=material_index(subset.GetPrim())
                for face in subset.GetIndicesAttr().Get(snapshot) or []:
                    if face<0 or face>=len(counts) or face in assigned: raise AssetError("Overlapping or invalid USD material subsets")
                    assigned.add(face);face_material[face]=index
            spherical_metadata=prim.GetCustomDataByKey("caustra:sphericalInteriorUV")
            if spherical_metadata is not None and not isinstance(spherical_metadata,bool):
                raise AssetError("caustra:sphericalInteriorUV must be a boolean")
            spherical_interior=bool(spherical_metadata)
            for index in set(face_material):
                if not asset.materials[index].program[3]&1: continue
                previous=interior_uv_mappings.setdefault(index,spherical_interior)
                if previous!=spherical_interior:
                    raise AssetError("A heterogeneous material cannot mix spherical and ordinary interior UV mappings; use distinct material bindings")
                if spherical_interior: asset.materials[index].program[3]|=2
            primvars=UsdGeom.PrimvarsAPI(prim)
            uv_data={}
            for index in set(face_material):
                names=compiler.material_uv_names.get(index,set())
                if len(names)>1: raise AssetError("A material uses more than one UV primvar; only one set is supported")
                name=next(iter(names),"st")
                uv=primvars.FindPrimvarWithInheritance(name)
                if names and not uv: raise AssetError("Material UV primvar is missing: "+name)
                uv_data[index]=(uv.ComputeFlattened(snapshot),str(uv.GetInterpolation())) if uv else (None,"constant")
            holes=set(mesh.GetHoleIndicesAttr().Get(snapshot) or [])
            grouped={};matrix=cache.GetLocalToWorldTransform(prim)
            left_handed=mesh.GetOrientationAttr().Get(snapshot)==UsdGeom.Tokens.leftHanded
            cursor=0
            for face,count in enumerate(counts):
                corners=list(indices[cursor:cursor+count])
                if face not in holes:
                    mat=face_material[face];target=grouped.setdefault(mat,Mesh(mat));base=len(target.vertices)
                    uv_values,uv_interpolation=uv_data[mat]
                    for local,vertex in enumerate(corners):
                        p=points[vertex]
                        normal=attr_value(normals,normal_interpolation,face,cursor+local,vertex,"normal")
                        uv=attr_value(uv_values,uv_interpolation,face,cursor+local,vertex,"UV")
                        if uv is not None and len(uv)!=2: raise AssetError("USD UV primvar must have two components")
                        target.vertices.append(list(p)+[0]+(list(normal) if normal is not None else [0,0,0])+[0]+(list(uv) if uv is not None else [0,0])+[0,0]+[0,0,0,0])
                        bounds.append(world_point(matrix,p))
                    for triangle in triangulate([points[i] for i in corners]):
                        a,b,c=triangle
                        if left_handed: b,c=c,b
                        target.indices.extend([base+a,base+b,base+c])
                cursor+=count
            for target in grouped.values():
                if target.indices:
                    displacement=compiler.material_displacements.get(target.material)
                    if displacement is not None:
                        from .displacement import displace_mesh
                        target=displace_mesh(target,displacement,asset.textures,transform(matrix),displacement_subdivisions)
                    asset.instances.append((len(asset.meshes),transform(matrix)));asset.meshes.append(target)
    # Geometry displacement can change every extent, so bounds are derived from
    # final object vertices and the actual world-meter instance transforms.
    bounds=[]
    for mesh_index,matrix in asset.instances:
        for vertex in asset.meshes[mesh_index].vertices:
            p=vertex[:3]
            bounds.append([sum(matrix[r*4+c]*p[c] for c in range(3))+matrix[r*4+3] for r in range(3)])
    if not asset.meshes: raise AssetError("USD scene has no visible supported meshes")
    asset.bounds_min=[min(p[i] for p in bounds) for i in range(3)]+[0]
    asset.bounds_max=[max(p[i] for p in bounds) for i in range(3)]+[0]
    center=[(a+b)*0.5 for a,b in zip(asset.bounds_min[:3],asset.bounds_max[:3])]
    if asset.has_camera and not camera_has_focus:
        position=asset.camera[:3];forward=[asset.camera[4+i]-position[i] for i in range(3)]
        length=math.sqrt(sum(x*x for x in forward))
        if length<=1e-20: raise AssetError("Degenerate USD camera transform")
        forward=[x/length for x in forward]
        distance=max(.1,sum((center[i]-position[i])*forward[i] for i in range(3)))
        asset.camera[4:7]=[position[i]+distance*forward[i] for i in range(3)]
    if not asset.has_camera:
        radius=max(0.1,math.dist(asset.bounds_min[:3],asset.bounds_max[:3])*0.5)
        position=[center[0]+radius*0.6,center[1]+radius*0.3,center[2]+radius*2.6]
        asset.camera=position+[0]+center+[0]+[0,1,0,0,45]
    return asset
