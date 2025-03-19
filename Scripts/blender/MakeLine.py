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

# Get all the vertices of the object, taking into account the object transformation
vertices = [obj.matrix_world @ v.co for v in obj.data.vertices]  # Apply the world transformation

# Sort the vertices based on the selected axis and direction
vertices.sort(key=lambda v: getattr(v, axis), reverse=reverse)

# Define the CSV file path in the same directory as the Blender file
csv_filename = os.path.join(blend_dir, "Start_Bend.csv")

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




# Optional: Print out the transformation matrix of the object
print("Object transformation matrix:")
print(obj.matrix_world)
