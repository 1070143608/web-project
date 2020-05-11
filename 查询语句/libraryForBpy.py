import bpy
def addMaterial(object, material):
    object.material_slots[0].material = bpy.data.materials[material].copy()

def changeLocation(object, location):
    object.location = tuple(location)
def changeRotation(object, rotation):
    object.rotation_euler = tuple(rotation)