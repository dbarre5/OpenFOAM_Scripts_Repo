import bpy
import csv
import os
import math
import random

# Path to your CSV file in the same directory as the .blend file
csv_filename = 'ripArea1_4.csv'
blend_dir = bpy.path.abspath("//")
csv_filepath = os.path.join(blend_dir, csv_filename)

def create_rock_mesh():
    # Randomize rings and segments between 3 and 5
    rings = random.randint(3, 5)
    segments = random.randint(3, 5)
    
    # Create a low-poly UV sphere with randomized rings and segments
    bpy.ops.mesh.primitive_uv_sphere_add(segments=segments, ring_count=rings, radius=0.75)
    rock = bpy.context.object
    rock.name = "BaseRock"
    
    # Enter Edit mode to modify vertices
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    
    # Apply random displacement to vertices
    #bpy.ops.transform.vertex_random(offset=0.2, uniform=0.1, normal=0.0, seed=random.randint(0, 100))
    
    bpy.ops.object.mode_set(mode='OBJECT')
    
    return rock

def read_csv_and_create_rocks(csv_filepath):
    vertices = []

    try:
        with open(csv_filepath, mode='r') as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in csv_reader:
                if len(row) != 3:
                    raise ValueError("CSV row does not contain exactly three values.")
                x, y, z = map(float, row)
                vertices.append((x, y, z))

    except ValueError as ve:
        print(f"ValueError: {ve}")
        return
    except FileNotFoundError:
        print("Error: CSV file not found.")
        return

    # Create a base rock with displaced vertices
    base_rock = create_rock_mesh()

    # Create a collection to store all rocks
    collection = bpy.data.collections.new("Rocks")
    bpy.context.scene.collection.children.link(collection)

    rocks = []

    for loc in vertices:
        # Create a copy of the base rock for each location
        rock = base_rock.copy()
        rock.data = base_rock.data.copy()
        bpy.context.collection.objects.link(rock)
        rock.location = loc

        # Apply random scaling
        scale_x = random.uniform(0.8, 1.2)
        scale_y = random.uniform(0.8, 1.2)
        scale_z = random.uniform(0.8, 1.2)
        rock.scale = (scale_x, scale_y, scale_z)

        # Apply random rotation
        rot_x = random.uniform(0, 2 * math.pi)
        rot_y = random.uniform(0, 2 * math.pi)
        rot_z = random.uniform(0, 2 * math.pi)
        rock.rotation_euler = (rot_x, rot_y, rot_z)

        collection.objects.link(rock)
        rocks.append(rock)

    # Join all rocks into one object
    if rocks:
        bpy.ops.object.select_all(action='DESELECT')
        for rock in rocks:
            rock.select_set(True)
        bpy.context.view_layer.objects.active = rocks[0]
        bpy.ops.object.join()

# Run the function
read_csv_and_create_rocks(csv_filepath)
