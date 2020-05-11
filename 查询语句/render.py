import bpy
import math
def rotate_render(output_dir):
    bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0, 0)) #create empty
    child = bpy.data.objects['Camera']
    parent = bpy.data.objects['Empty']
    child.select_set(True) # choose child
    parent.select_set(True) #choose parent
    bpy.ops.object.parent_set() 
    bpy.ops.object.select_all(action='DESELECT') #deselect all object
    
    #output_dir = "C:\\Users\\86139\\Desktop\\"
    for i in range(0, 10):
        parent.rotation_euler[2] = parent.rotation_euler[2] + 2 * math.pi / 15
        bpy.context.scene.render.filepath = output_dir + str(i + 1)
        bpy.ops.render.render(write_still = True)
        print(i)
def main():
    rotate_render("C:\\Users\\boom\\Desktop\\xing\\test\\")
main()