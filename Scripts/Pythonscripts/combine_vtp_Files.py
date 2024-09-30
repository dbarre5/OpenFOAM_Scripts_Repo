import os
import vtk

# User-defined variables
base_folder = 'postProcessing'
sub_folder_name = 'isosurfaceAlphaWater'
output_folder = 'waterSurfaceData'
vertical_direction = 'z'  # Change this to 'x' or 'y' if needed

def read_vtp_file(filename):
    """Reads a .vtp file and returns the vtkPolyData."""
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()

def write_vtp_file(polydata, filename):
    """Writes the vtkPolyData to a .vtp file."""
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(filename)
    writer.SetInputData(polydata)
    writer.Write()

def add_vertical_coordinate_dataset(polydata, vertical_direction):
    """Adds a new dataset to the polydata with values equal to the vertical direction coordinates of the points."""
    points = polydata.GetPoints()
    vertical_values = vtk.vtkDoubleArray()
    vertical_values.SetName(f"{vertical_direction}Values")
    vertical_values.SetNumberOfComponents(1)
    vertical_values.SetNumberOfTuples(points.GetNumberOfPoints())

    for i in range(points.GetNumberOfPoints()):
        coords = points.GetPoint(i)
        vertical_value = coords["xyz".index(vertical_direction)]
        vertical_values.SetValue(i, vertical_value)
    
    polydata.GetPointData().AddArray(vertical_values)

def find_vtp_files(base_folder, sub_folder_name):
    """Finds all .vtp files in the simulation time subfolders."""
    vtp_files = []
    sub_folder_path = os.path.join(base_folder, sub_folder_name)
    
    if os.path.isdir(sub_folder_path):
        for time_folder in os.listdir(sub_folder_path):
            time_folder_path = os.path.join(sub_folder_path, time_folder)
            if os.path.isdir(time_folder_path):
                for file in os.listdir(time_folder_path):
                    if file.endswith('.vtp'):
                        vtp_files.append((os.path.join(time_folder_path, file), float(time_folder)))
    return sorted(vtp_files, key=lambda x: x[1])

def write_pvd_file(file_list, output_pvd_file):
    """Writes a .pvd file that references all the .vtp files."""
    with open(output_pvd_file, 'w') as pvd:
        pvd.write('<?xml version="1.0"?>\n')
        pvd.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
        pvd.write('  <Collection>\n')
        for filename, time_value in file_list:
            pvd.write(f'    <DataSet timestep="{time_value}" group="" part="0" file="{filename}" />\n')
        pvd.write('  </Collection>\n')
        pvd.write('</VTKFile>\n')


# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Find all .vtp files
vtp_files = find_vtp_files(base_folder, sub_folder_name)

# Process and write each .vtp file with time step
output_vtp_files = []
for idx, (filename, time_value) in enumerate(vtp_files):
    polydata = read_vtp_file(filename)
    add_vertical_coordinate_dataset(polydata, vertical_direction)
    output_vtp_filename = os.path.join(output_folder, f'timestep_{idx:04d}.vtp')
    write_vtp_file(polydata, output_vtp_filename)
    output_vtp_files.append((output_vtp_filename, time_value))

# Write .pvd file
output_pvd_file = os.path.join(os.getcwd(), 'waterSurface.pvd')  # Output .pvd file in the current directory
write_pvd_file(output_vtp_files, output_pvd_file)

print(f"Processed {len(vtp_files)} .vtp files and wrote {output_pvd_file}")