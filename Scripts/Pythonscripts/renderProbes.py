import os

point_size_ratio = 0.2
STL_opacity = 0.3
Probe_number_size = 0
STL_Edge_width = 3

def extract_probe_locations(directory, file_name):
    """
    Extract probe locations from files in the specified directory.
    
    Parameters:
    directory (str): The root directory to search for probe files.
    file_name (str): The specific file name to look for in the directories.
    
    Returns:
    dict: A dictionary where keys are group names (parent directory names)
          and values are lists of (x, y, z) probe coordinates.
    """
    all_probe_data = {}

    for root, _, files in os.walk(directory):
        if file_name in files:
            probe_data = []
            file_path = os.path.join(root, file_name)
            with open(file_path, 'r') as f:
                lines = f.readlines()

            for line in lines:
                if line.startswith("# Probe"):
                    parts = line.split()
                    x = float(parts[3][1:])
                    y = float(parts[4])
                    z = float(parts[5][0:-1])
                    probe_data.append((x, y, z))

            group_name = os.path.basename(os.path.dirname(root))
            all_probe_data[group_name] = probe_data

    return all_probe_data

# Example usage
directory = "postProcessing"
file_name = 'multU'  # or any other file name you're looking for
probe_locations = extract_probe_locations(directory, file_name)

# Define the directory containing the STL files
stl_directory = "constant/triSurface/"

# Function to check for VTK installation
try:
    import vtk
    vtk_installed = True
except ImportError:
    vtk_installed = False

# Search for STL files within the directory
stl_files = [os.path.join(stl_directory, file) for file in os.listdir(stl_directory) if file.endswith('.stl')]

if vtk_installed:
    # Function to calculate the total length of the domain defined by the STL files
    def calculate_domain_length(stl_files):
        min_bound = [float('inf'), float('inf'), float('inf')]
        max_bound = [-float('inf'), -float('inf'), -float('inf')]
        
        for stl_file in stl_files:
            reader = vtk.vtkSTLReader()
            reader.SetFileName(stl_file)
            reader.Update()
            
            bounds = reader.GetOutput().GetBounds()
            min_bound = [min(min_bound[i], bounds[2*i]) for i in range(3)]
            max_bound = [max(max_bound[i], bounds[2*i+1]) for i in range(3)]
        
        total_domain_length = max(max_bound[i] - min_bound[i] for i in range(3))
        return total_domain_length

    # Calculate the total length of the domain defined by the STL files
    total_domain_length = calculate_domain_length(stl_files)

    # Define the desired length scale for the probe points
    desired_probe_length_scale = 0.01*point_size_ratio  # Adjust this value as needed

    # Calculate the scaling factor for the probe points
    probe_scaling_factor = total_domain_length * desired_probe_length_scale

    # Function to load and render STL files and points using VTK
    def render_with_vtk(stl_files, x_points, y_points, z_points, output_filename="isometric_view.png"):
        # Create a rendering window and renderer
        ren = vtk.vtkRenderer()
        renWin = vtk.vtkRenderWindow()
        renWin.AddRenderer(ren)

        # Set the size of the render window (increase size for higher resolution)
        renWin.SetSize(1920, 1920)  # For example, 1920x1920 for higher resolution

        
        # Load the STL files
        for stl_file in stl_files:
            reader = vtk.vtkSTLReader()
            reader.SetFileName(stl_file)
            reader.Update()

            polydata = reader.GetOutput()

            # Create an edge array to store the edges
            edge_array = {}

            # Iterate over the cells (triangles)
            for cell_id in range(polydata.GetNumberOfCells()):
                cell = polydata.GetCell(cell_id)

                # Iterate over the edges of the cell
                for edge_id in range(cell.GetNumberOfEdges()):
                    edge = cell.GetEdge(edge_id)
                    edge_tuple = tuple(sorted([edge.GetPointId(0), edge.GetPointId(1)]))

                    # If the edge exists in the edge array, remove it
                    if edge_tuple in edge_array:
                        del edge_array[edge_tuple]
                    else:
                        edge_array[edge_tuple] = edge_tuple

            # Create the mapper for edges
            lines_mapper = vtk.vtkPolyDataMapper()

            # Create the polydata for the edges
            lines_polydata = vtk.vtkPolyData()
            lines_points = vtk.vtkPoints()
            lines_cells = vtk.vtkCellArray()

            # Add the points to the polydata
            for edge_tuple in edge_array.values():
                point1 = polydata.GetPoint(edge_tuple[0])
                point2 = polydata.GetPoint(edge_tuple[1])

                lines_points.InsertNextPoint(point1)
                lines_points.InsertNextPoint(point2)

                line = vtk.vtkLine()
                line.GetPointIds().SetId(0, lines_points.GetNumberOfPoints() - 2)
                line.GetPointIds().SetId(1, lines_points.GetNumberOfPoints() - 1)

                lines_cells.InsertNextCell(line)

            lines_polydata.SetPoints(lines_points)
            lines_polydata.SetLines(lines_cells)

            lines_mapper.SetInputData(lines_polydata)

            # Create the actor for edges
            lines_actor = vtk.vtkActor()
            lines_actor.SetMapper(lines_mapper)
            # Set the line width in the actor's property
            lines_actor.GetProperty().SetLineWidth(STL_Edge_width)  # Set line width to 2 (adjust as needed)

            lines_actor.GetProperty().SetColor(0, 0, 0)  # Black color

            ren.AddActor(lines_actor)
            # Now render the surface without internal boundary lines
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(reader.GetOutputPort())

            actor = vtk.vtkActor()
            actor.SetMapper(mapper)
            actor.GetProperty().SetOpacity(STL_opacity)  # Set 30% opacity

            ren.AddActor(actor)   


        # Plot the points with scaling
        points = vtk.vtkPoints()
        for x, y, z in zip(x_points, y_points, z_points):
            points.InsertNextPoint(x, y, z)

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)

        vertexGlyphFilter = vtk.vtkVertexGlyphFilter()
        vertexGlyphFilter.SetInputData(polydata)
        vertexGlyphFilter.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(vertexGlyphFilter.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetColor(1, 0, 0)  # Red color

        # Add spheres to represent probe points
        sphere = vtk.vtkSphereSource()
        sphere.SetRadius(probe_scaling_factor)  # Set the radius to the scaling factor
        sphere_mapper = vtk.vtkPolyDataMapper()
        sphere_mapper.SetInputConnection(sphere.GetOutputPort())

        for x, y, z in zip(x_points, y_points, z_points):
            sphere_actor = vtk.vtkActor()
            sphere_actor.SetMapper(sphere_mapper)
            sphere_actor.SetPosition(x, y, z)
            sphere_actor.GetProperty().SetColor(1, 0, 0)  # Red color
            ren.AddActor(sphere_actor)

        # Add text labels for probe indices
        for i, (x, y, z) in enumerate(zip(x_points, y_points, z_points)):
            text = vtk.vtkVectorText()
            text.SetText(str(i))

            text_mapper = vtk.vtkPolyDataMapper()
            text_mapper.SetInputConnection(text.GetOutputPort())

            text_actor = vtk.vtkFollower()
            text_actor.SetMapper(text_mapper)
            text_actor.SetPosition(x + probe_scaling_factor, y + probe_scaling_factor, z + probe_scaling_factor)
            text_actor.SetScale(probe_scaling_factor*Probe_number_size)
            text_actor.GetProperty().SetColor(0, 0, 0)  # Black color
            text_actor.SetCamera(ren.GetActiveCamera())

            ren.AddActor(text_actor)

        # Set the background color to white
        ren.SetBackground(1, 1, 1)

        # Set up the camera for an isometric view
        camera = ren.GetActiveCamera()
        camera.ParallelProjectionOn()
        camera.SetPosition(1, 1, 1)
        camera.SetFocalPoint(0, 0, 0)
        camera.SetViewUp(0, 0, 1)
        ren.ResetCamera()
        # Instead of rendering to the window, render off-screen
        renWin.SetOffScreenRendering(1)

        # Render the scene
        renWin.Render()

        # Save the render window to an image
        window_to_image_filter = vtk.vtkWindowToImageFilter()
        window_to_image_filter.SetInput(renWin)
        window_to_image_filter.SetScale(4)  # Increase scale for higher DPI
        window_to_image_filter.Update()

        writer = vtk.vtkPNGWriter()
        writer.SetFileName(output_filename)
        writer.SetInputConnection(window_to_image_filter.GetOutputPort())
        writer.Write()

    # Loop over each group of probe locations and render them
    for group_name, probes in probe_locations.items():
        if probes:
            x_points, y_points, z_points = zip(*probes)
        else:
            x_points, y_points, z_points = [], [], []

        output_filename = f"{group_name}_isometric_view.png"
        render_with_vtk(stl_files, x_points, y_points, z_points, output_filename)
else:
    print("VTK is not installed. Please install VTK using 'pip install vtk' to render the STL files and points.")