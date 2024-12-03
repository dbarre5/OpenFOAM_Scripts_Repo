import bpy
import csv
import os
import numpy as np

# Get the path to the current Blender file
blend_file_path = bpy.data.filepath
if blend_file_path == "":
    raise ValueError("The Blender file is not saved yet. Please save the Blender file first.")

# Get the directory of the Blender file
blend_dir = os.path.dirname(blend_file_path)

# Step 1: Edit the sorting axis and direction
axis = 'x'      # Choose between 'x', 'y', 'z'
reverse = True   # True for negative direction, False for positive direction

# Step 2: Output points of the object (line) into a CSV
obj = bpy.context.object

# Make sure the object is in Object mode
bpy.ops.object.mode_set(mode='OBJECT')

# Get all the vertices of the object
vertices = [v.co for v in obj.data.vertices]

# Sort the vertices based on the selected axis and direction
vertices.sort(key=lambda v: getattr(v, axis), reverse=reverse)

# Define the CSV file path in the same directory as the Blender file
csv_filename = os.path.join(blend_dir, "line_points.csv")

# Write to CSV (without column names)
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    for vertex in vertices:
        writer.writerow([vertex.x, vertex.y, vertex.z])

print(f"CSV written to {csv_filename}")

# Step 3: Visualize the points in Blender 3D Viewport
# Convert to numpy array for easier manipulation
points = np.array([[v.x, v.y, v.z] for v in vertices])

# Calculate the total length of the line
distances = np.linalg.norm(np.diff(points, axis=0), axis=1)  # Euclidean distance between consecutive points
total_length = np.sum(distances)

# Calculate the percentage distance along the line
cumulative_distances = np.cumsum(distances)
percentage_distances = cumulative_distances / total_length

# Create a new collection for the visual points
collection = bpy.data.collections.new("Visualized Points")
bpy.context.scene.collection.children.link(collection)

# Define the RGB values for the inferno colormap based on the percentage (using your provided values)
inferno_colors = [
    (0.001462, 0.000466, 0.013866), (0.003299, 0.002249, 0.024239), (0.007676, 0.006136, 0.046836),
    (0.011663, 0.009417, 0.06346), (0.019373, 0.015133, 0.088767), (0.025793, 0.019331, 0.10593),
    (0.037668, 0.025921, 0.132232), (0.051644, 0.032474, 0.159254), (0.06134, 0.03659, 0.177642),
    (0.076637, 0.041905, 0.205799), (0.087411, 0.044556, 0.224813), (0.104551, 0.047008, 0.25343),
    (0.122908, 0.047536, 0.281624), (0.135778, 0.046856, 0.299776), (0.15585, 0.044559, 0.325338),
    (0.169575, 0.042489, 0.340874), (0.190367, 0.039309, 0.361447), (0.204209, 0.037632, 0.373238),
    (0.224763, 0.036405, 0.388129), (0.244967, 0.037055, 0.400007), (0.258234, 0.038571, 0.406485),
    (0.27785, 0.042353, 0.414392), (0.290763, 0.045644, 0.418637), (0.309935, 0.051407, 0.423721),
    (0.328921, 0.057827, 0.427511), (0.3415, 0.062325, 0.429425), (0.360284, 0.069247, 0.431497),
    (0.372768, 0.073915, 0.4324), (0.391453, 0.080927, 0.433109), (0.403894, 0.08558, 0.433179),
    (0.422549, 0.092501, 0.432714), (0.441207, 0.099338, 0.431594), (0.453651, 0.103848, 0.430498),
    (0.472328, 0.110547, 0.428334), (0.484789, 0.114974, 0.426548), (0.503493, 0.121575, 0.423356),
    (0.522206, 0.12815, 0.419549), (0.534683, 0.132534, 0.416667), (0.553392, 0.139134, 0.411829),
    (0.565854, 0.143567, 0.408258), (0.584521, 0.150294, 0.402385), (0.603139, 0.157151, 0.395891),
    (0.615513, 0.161817, 0.391219), (0.633998, 0.168992, 0.383704), (0.64626, 0.173914, 0.378359),
    (0.66454, 0.181539, 0.369846), (0.676638, 0.186807, 0.363849), (0.694627, 0.195021, 0.354388),
    (0.712396, 0.203656, 0.344383), (0.724103, 0.20967, 0.337424), (0.741423, 0.219112, 0.326576),
    (0.752794, 0.225706, 0.319085), (0.769556, 0.236077, 0.307485), (0.785929, 0.247056, 0.295477),
    (0.796607, 0.254728, 0.287264), (0.812239, 0.266786, 0.274661), (0.822386, 0.275197, 0.266085),
    (0.837165, 0.288385, 0.252988), (0.846709, 0.297559, 0.244113), (0.860533, 0.311892, 0.230606),
    (0.873741, 0.326906, 0.216886), (0.882188, 0.337287, 0.207628), (0.894305, 0.353399, 0.193584),
    (0.902003, 0.364492, 0.184116), (0.912966, 0.381636, 0.169755), (0.923215, 0.399359, 0.155193),
    (0.929644, 0.411479, 0.145367), (0.938675, 0.430091, 0.130438), (0.944285, 0.442772, 0.120354),
    (0.952075, 0.462178, 0.105031), (0.959114, 0.482014, 0.089499), (0.963387, 0.495462, 0.079073),
    (0.969163, 0.515946, 0.063488), (0.97259, 0.529798, 0.053324), (0.977092, 0.55085, 0.03905),
    (0.979666, 0.565057, 0.031409), (0.982881, 0.586606, 0.024661), (0.985315, 0.608422, 0.024202),
    (0.986502, 0.623105, 0.027814), (0.988096, 0.643052, 0.035868), (0.988162, 0.657396, 0.042264),
    (0.989293, 0.673724, 0.049279), (0.987714, 0.682807, 0.072489)
]

# Function to apply Emission node with color
def create_material_with_emission(color):
    # Create a new material
    mat = bpy.data.materials.new(name="Emission_Material")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes

    # Clear default nodes
    for node in nodes:
        nodes.remove(node)

    # Add an Emission node
    emission_node = nodes.new(type='ShaderNodeEmission')
    emission_node.inputs["Color"].default_value = (*color, 1)  # RGBA, with alpha=1
    emission_node.inputs["Strength"].default_value = 0.5  # Set the emission strength

    # Add a Material Output node and link the emission node to it
    material_output_node = nodes.new(type='ShaderNodeOutputMaterial')
    mat.node_tree.links.new(emission_node.outputs[0], material_output_node.inputs[0])

    return mat

# Visualize each point as a sphere with color based on its percentage distance along the line
for i, (point, percentage) in enumerate(zip(points, percentage_distances)):
    # Create a sphere for each point
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.1, location=point)
    sphere = bpy.context.object

    # Get the RGB color based on the percentage from the inferno colormap (use your list of colors here)
    color = inferno_colors[int(percentage * (len(inferno_colors) - 1))]

    # Create material with Emission and set the color
    mat = create_material_with_emission(color)

    # Assign material to the sphere
    sphere.data.materials.append(mat)

    # Link the sphere to the created collection
    collection.objects.link(sphere)


# Define the percentage markers
percentage_markers = [0, 0.2, 0.4, 0.6, 0.8, 1.0]


# Function to add text object at a specific location
def add_text_marker(text, location, size=0.2):
    # Add text object at the specified location (on the xy-plane)
    bpy.ops.object.text_add(location=(location[0], location[1], 0))  # Placing text on the xy-plane
    text_obj = bpy.context.object
    text_obj.data.body = text  # Set the text content
    
    # Adjust the rotation so the text faces the top-down view (xy-plane)
    text_obj.rotation_euler[2] = np.pi / 2  # Rotate around Z to face the camera top-down
    
    # Set the text size
    text_obj.data.size = size
    
    # Create a rough, non-reflective black material and assign it to the text
    dark_matter_material = bpy.data.materials.new(name="Dark_Matter_Material")
    dark_matter_material.use_nodes = True
    bsdf = dark_matter_material.node_tree.nodes["Principled BSDF"]
    
    # Set black color (RGBA)
    bsdf.inputs["Base Color"].default_value = (0, 0, 0, 1)
    
    # Make it rough and non-reflective
    bsdf.inputs["Roughness"].default_value = 1.0  # Make it matte
    bsdf.inputs["Specular"].default_value = 0.0  # No reflections
    
    # Optional: Add faint emission to make it glow slightly like dark matter (optional)
    # bsdf.inputs["Emission"].default_value = (0, 0, 0, 0)  # Keep emission off, but you could uncomment this line and add a faint glow like (0.1, 0.1, 0.1, 1)
    
    # Assign the dark matter material to the text object
    if len(text_obj.data.materials) == 0:
        text_obj.data.materials.append(dark_matter_material)
    else:
        text_obj.data.materials[0] = dark_matter_material
    
    return text_obj

# Get the points (already calculated in your previous code as 'points' and 'percentage_distances')
# Select the appropriate percentage positions
for percentage in percentage_markers:
    # Find the point corresponding to the current percentage
    index = int(percentage * (len(percentage_distances) - 1))
    point = points[index]
    
    # Add the text marker at the corresponding point with a specific size (e.g., size=1.0)
    add_text_marker(f"{int(percentage * 100)}%", point, size=0.2)

print("Text markers with matte, dark matter-like appearance added.")

# Optional: Align view to see the markers, but you can skip this if not required
bpy.ops.view3d.view_all(center=True)