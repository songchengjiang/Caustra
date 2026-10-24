"""Reproducible feature gallery, authored here without downloaded models.

1. blender --background --python tools/blender_export_features.py -- tests/assets/features/blender_native.usdc
2. python tools/blender_export_features.py --augment tests/assets/features/blender_native.usdc

The second step keeps Blender's native OpenPBR networks and explicitly authors
MaterialX volume/displacement terminals that Blender 4.5 does not export.
"""
import math
import sys
from pathlib import Path


def export_blender(output):
    import bpy
    from mathutils import Vector
    bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
    def material(name,color,roughness=.22):
        m=bpy.data.materials.new(name);m.use_nodes=True;p=m.node_tree.nodes.get('Principled BSDF')
        p.inputs['Base Color'].default_value=(*color,1);p.inputs['Roughness'].default_value=roughness
        return m,p
    specs=[('Glass_Clear',(1,1,1),.04),('Glass_Tinted',(.15,.65,.95),.08),
           ('Glass_Frosted',(.9,.95,1),.38),('Glass_ThinWalled',(.95,.65,.25),.08),
           ('Subsurface_Wax',(.8,.22,.08),.35),('Wool_Fuzz',(.16,.35,.55),.55),
           ('ThinFilm',(.65,.65,.65),.18),('Volume_Box',(.5,.5,.5),.3)]
    for i,(name,color,roughness) in enumerate(specs):
        x=(i%4-1.5)*2.2;y=(i//4)*2.7
        mat,p=material(name+'_Material',color,roughness)
        if name.startswith('Glass_'): p.inputs['Transmission Weight'].default_value=1
        if name=='Subsurface_Wax':
            p.inputs['Subsurface Weight'].default_value=.9;p.inputs['Subsurface Radius'].default_value=(1,.35,.15)
            p.inputs['Subsurface Scale'].default_value=.3
        if name=='Wool_Fuzz':
            p.inputs['Sheen Weight'].default_value=.9;p.inputs['Sheen Roughness'].default_value=.6
        if name=='ThinFilm':
            p.inputs['Metallic'].default_value=.75;p.inputs['Thin Film Thickness'].default_value=450
            p.inputs['Thin Film IOR'].default_value=1.4
        if name=='Volume_Box':
            bpy.ops.mesh.primitive_cube_add(size=1.6,location=(x,y,.9))
        else:
            bpy.ops.mesh.primitive_uv_sphere_add(segments=32,ring_count=16,radius=.85,location=(x,y,1))
        obj=bpy.context.object;obj.name=name;obj.data.materials.append(mat)
        for f in obj.data.polygons: f.use_smooth=name!='Volume_Box'
    mat,p=material('Displacement_Material',(.38,.18,.07),.4)
    bpy.ops.mesh.primitive_grid_add(x_subdivisions=5,y_subdivisions=5,size=2.2,location=(0,-2.5,.25))
    obj=bpy.context.object;obj.name='Displaced_Mesh';obj.data.materials.append(mat)
    floor,p=material('Ground_Material',(.19,.21,.24),.7)
    bpy.ops.mesh.primitive_plane_add(size=35);bpy.context.object.name='Ground';bpy.context.object.data.materials.append(floor)
    for name,loc,emission,size in [('Key',(-3,-3,7),(13,12,10),5),('Fill',(4,3,6),(7,9,14),4)]:
        mat,p=material(name+'_Emitter_Material',(0,0,0));p.inputs['Emission Color'].default_value=(*emission,1)
        p.inputs['Emission Strength'].default_value=1
        bpy.ops.mesh.primitive_plane_add(size=size,location=loc);obj=bpy.context.object;obj.name=name+'_Emitter'
        obj.rotation_euler=(Vector((0,1,1))-obj.location).to_track_quat('Z','Y').to_euler();obj.data.materials.append(mat)
    bpy.ops.object.camera_add(location=(8,-12,9));camera=bpy.context.object
    camera.rotation_euler=(Vector((0,.6,1))-camera.location).to_track_quat('-Z','Y').to_euler()
    camera.data.lens=48;bpy.context.scene.camera=camera
    output.parent.mkdir(parents=True,exist_ok=True)
    bpy.context.preferences.filepaths.save_version=0
    bpy.ops.wm.save_as_mainfile(filepath=str(output.parent/'scene.blend'))
    bpy.ops.wm.usd_export(filepath=str(output),generate_materialx_network=True,generate_preview_surface=False,
                          export_textures=True,relative_paths=True,triangulate_meshes=True,
                          export_animation=False,use_instancing=False,convert_world_material=False)
    print('CAUSTRA_FEATURE_EXPORT',output)


def augment(native):
    from pxr import Sdf,Usd,UsdShade,UsdGeom,Gf
    import MaterialX as mx
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    from caustra_assets.materialx_compiler import standard_library
    lib=standard_library();output=native.parent/'scene.usda'
    stage=Usd.Stage.CreateNew(str(output));stage.GetRootLayer().subLayerPaths=[native.name]
    source=Usd.Stage.Open(str(native))
    UsdGeom.SetStageMetersPerUnit(stage,UsdGeom.GetStageMetersPerUnit(source))
    UsdGeom.SetStageUpAxis(stage,UsdGeom.GetStageUpAxis(source))
    mats={p.GetName():UsdShade.Material(p) for p in stage.Traverse() if p.IsA(UsdShade.Material)}
    def shader(material,name,identifier):
        node=UsdShade.Shader.Define(stage,str(material.GetPath())+'/'+name);node.CreateIdAttr(identifier)
        node.CreateOutput('out',Sdf.ValueTypeNames.Token);return node
    thin=mats['Glass_ThinWalled_Material'];surface=thin.ComputeSurfaceSource('mtlx')[0]
    surface.CreateInput('geometry_thin_walled',Sdf.ValueTypeNames.Bool).Set(True)
    # Blender currently exports clear transmission; add an authored physical
    # distance for the tinted test so this is attenuation inside a solid.
    tinted=mats['Glass_Tinted_Material'];surface=tinted.ComputeSurfaceSource('mtlx')[0]
    surface.CreateInput('transmission_depth',Sdf.ValueTypeNames.Float).Set(.7)
    volume=mats['Volume_Box_Material'];volume.GetSurfaceOutput('mtlx').DisconnectSource()
    vdf=shader(volume,'caustra_vdf','ND_anisotropic_vdf')
    vdf.CreateInput('absorption',Sdf.ValueTypeNames.Vector3f).Set(Gf.Vec3f(.15,.04,.02))
    vdf.CreateInput('scattering',Sdf.ValueTypeNames.Vector3f).Set(Gf.Vec3f(.55,.65,.8))
    vdf.CreateInput('anisotropy',Sdf.ValueTypeNames.Float).Set(.35)
    edf=shader(volume,'caustra_edf','ND_uniform_edf');edf.CreateInput('color',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.01,.015,.025))
    terminal=shader(volume,'caustra_volume','ND_volume')
    terminal.CreateInput('vdf',Sdf.ValueTypeNames.Token).ConnectToSource(vdf.ConnectableAPI(),'out')
    terminal.CreateInput('edf',Sdf.ValueTypeNames.Token).ConnectToSource(edf.ConnectableAPI(),'out')
    volume.CreateVolumeOutput('mtlx').ConnectToSource(terminal.ConnectableAPI(),'out')
    material=mats['Displacement_Material'];position=shader(material,'caustra_position','ND_position_vector3')
    position.CreateInput('space',Sdf.ValueTypeNames.String).Set('object');position.CreateOutput('out',Sdf.ValueTypeNames.Float3)
    noise=shader(material,'caustra_noise','ND_noise3d_float');noise.CreateOutput('out',Sdf.ValueTypeNames.Float)
    noise.CreateInput('position',Sdf.ValueTypeNames.Vector3f).ConnectToSource(position.ConnectableAPI(),'out')
    noise.CreateInput('amplitude',Sdf.ValueTypeNames.Float).Set(.4);noise.CreateInput('pivot',Sdf.ValueTypeNames.Float).Set(.1)
    disp=shader(material,'caustra_displacement','ND_displacement_float')
    disp.CreateInput('displacement',Sdf.ValueTypeNames.Float).ConnectToSource(noise.ConnectableAPI(),'out')
    material.CreateDisplacementOutput('mtlx').ConnectToSource(disp.ConnectableAPI(),'out')
    stage.GetRootLayer().Save()
    # Paired editable, standards-valid MaterialX source for auxiliary terminals.
    for material,name in ((volume,'volume'),(material,'displacement')):
        from caustra_assets.usd_import import materialx_from_usd
        document,_=materialx_from_usd(material,lib)
        mx.writeToXmlFile(document,str(native.parent/(name+'.mtlx')))
    print('CAUSTRA_FEATURE_AUGMENT',output)


if __name__=='__main__':
    if '--augment' in sys.argv:
        augment(Path(sys.argv[sys.argv.index('--augment')+1]).resolve())
    else:
        args=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else []
        export_blender(Path(args[0] if args else 'tests/assets/features/blender_native.usdc').resolve())
