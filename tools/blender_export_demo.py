"""Run with Blender --background --python tools/blender_export_demo.py -- output.usda.

Creates a reproducible USD asset with Blender's native MaterialX exporter.
The geometry and textures are authored by this script; no external assets.
"""
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def material(name, color, metal=0.0, roughness=0.35):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    shader = mat.node_tree.nodes.get('Principled BSDF')
    shader.inputs['Base Color'].default_value = (*color, 1.0)
    shader.inputs['Metallic'].default_value = metal
    shader.inputs['Roughness'].default_value = roughness
    return mat, shader


def image(path, name, normal=False):
    width = height = 64
    img = bpy.data.images.new(name, width, height, alpha=True)
    img.colorspace_settings.name = 'Non-Color' if normal else 'sRGB'
    pixels = []
    for y in range(height):
        for x in range(width):
            if normal:
                nx = 0.18 * math.sin(x * math.pi / 4)
                ny = 0.18 * math.sin(y * math.pi / 4)
                nz = math.sqrt(1 - nx * nx - ny * ny)
                pixels.extend((0.5 + nx / 2, 0.5 + ny / 2, 0.5 + nz / 2, 1.0))
            else:
                c = (0.08, 0.3, 0.75) if ((x // 8) + (y // 8)) % 2 else (0.95, 0.72, 0.12)
                pixels.extend((*c, 1.0))
    img.pixels = pixels
    img.filepath_raw = str(path)
    img.file_format = 'PNG'
    img.save()
    return img


def main():
    arguments = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
    output = Path(arguments[0] if arguments else 'renders/blender_materialx.usda').resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

    paint, shader = material('MaterialX_Checker_Coat', (0.2, 0.4, 0.8))
    shader.inputs['Coat Weight'].default_value = 0.35
    shader.inputs['Coat Roughness'].default_value = 0.18
    texture = paint.node_tree.nodes.new('ShaderNodeTexImage')
    texture.image = image(output.parent / 'checker.png', 'CheckerColor')
    paint.node_tree.links.new(texture.outputs['Color'], shader.inputs['Base Color'])
    normal_texture = paint.node_tree.nodes.new('ShaderNodeTexImage')
    normal_texture.image = image(output.parent / 'normal.png', 'NormalData', normal=True)
    normal_map = paint.node_tree.nodes.new('ShaderNodeNormalMap')
    normal_map.inputs['Strength'].default_value = 0.55
    paint.node_tree.links.new(normal_texture.outputs['Color'], normal_map.inputs['Color'])
    paint.node_tree.links.new(normal_map.outputs['Normal'], shader.inputs['Normal'])
    bpy.ops.mesh.primitive_uv_sphere_add(segments=48, ring_count=24, radius=0.95, location=(-1.15, 0, 1.1))
    sphere = bpy.context.object
    sphere.name = 'Textured_Sphere'
    sphere.data.materials.append(paint)
    for face in sphere.data.polygons:
        face.use_smooth = True

    copper, shader = material('MaterialX_Anisotropic_Copper', (0.8, 0.35, 0.12), 1.0, 0.24)
    shader.inputs['Anisotropic'].default_value = 0.45
    bpy.ops.mesh.primitive_torus_add(major_segments=64, minor_segments=24, major_radius=0.78,
                                   minor_radius=0.24, location=(1.15, 0, 1.15),
                                   rotation=(math.pi / 2, 0.2, 0.1))
    torus = bpy.context.object
    torus.name = 'Anisotropic_Copper'
    torus.data.materials.append(copper)
    for face in torus.data.polygons:
        face.use_smooth = True

    floor, _ = material('MaterialX_Ground', (0.18, 0.2, 0.23), 0.0, 0.65)
    bpy.ops.mesh.primitive_plane_add(size=30)
    bpy.context.object.name = 'Ground'
    bpy.context.object.data.materials.append(floor)

    # Emissive geometry exercises the same MaterialX graph in NEE and ray hits.
    for name, location, size, emission in (
        ('Key_Emitter', (-3, -2, 5), 3.0, (12.0, 11.0, 9.0)),
        ('Fill_Emitter', (3, 1, 4), 2.0, (5.0, 7.0, 12.0)),
    ):
        lamp, shader = material(name + '_Material', (0.0, 0.0, 0.0))
        shader.inputs['Emission Color'].default_value = (*emission, 1.0)
        shader.inputs['Emission Strength'].default_value = 1.0
        bpy.ops.mesh.primitive_plane_add(size=size, location=location)
        obj = bpy.context.object
        obj.name = name
        obj.rotation_euler = (Vector((0, 0, 1)) - obj.location).to_track_quat('Z', 'Y').to_euler()
        obj.data.materials.append(lamp)

    bpy.ops.object.camera_add(location=(5.2, -8.5, 4.0))
    camera = bpy.context.object
    camera.rotation_euler = (Vector((0, 0, 1.0)) - camera.location).to_track_quat('-Z', 'Y').to_euler()
    camera.data.lens = 48
    bpy.context.scene.camera = camera
    bpy.context.scene.render.resolution_x = 960
    bpy.context.scene.render.resolution_y = 640
    bpy.ops.wm.usd_export(filepath=str(output), generate_materialx_network=True,
                          generate_preview_surface=False, export_textures=True,
                          overwrite_textures=True, relative_paths=True,
                          triangulate_meshes=True, export_animation=False,
                          use_instancing=False, convert_world_material=False)
    print('CAUSTRA_BLENDER_EXPORT', output)


if __name__ == '__main__':
    main()
