import bpy
scn = bpy.context.scene
for f in range(scn.frame_start, scn.frame_end):
    fpath = bpy.path.abspath('//waterSurfaceStar0_{}.stl'.format(f))
    bpy.ops.import_mesh.stl(filepath=fpath)
    obj = bpy.context.active_object
    mat = bpy.data.materials.get("Material.002")
    obj.data.materials.append(mat)
    # key as visible on the current frame
    obj.keyframe_insert('hide_viewport',frame=f)
    obj.keyframe_insert('hide_render',frame=f)
    # hide it
    obj.hide_viewport = True
    obj.hide_render = True
    # key as hidden on the previous frame
    obj.keyframe_insert('hide_viewport',frame=f-1)
    obj.keyframe_insert('hide_render',frame=f-1)
    # key as hidden on the next frame
    obj.keyframe_insert('hide_viewport',frame=f+1)
    obj.keyframe_insert('hide_render',frame=f+1)
