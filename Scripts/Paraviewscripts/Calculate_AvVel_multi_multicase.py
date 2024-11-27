

import os
from paraview.simple import *
import numpy as np
import matplotlib.pyplot as plt
import time

cases = ["mesh1", "mesh2"]

# User inputs vertical direction: 0 for x, 1 for y, 2 for z
vertical_direction = 2



# Specify the desired time step (e.g., 'latest' or a specific time like 10.0)
desired_time = 2000

########################## GRID OPTIONS
grid = False
# Specify bounds of your domain (min_x, max_x, min_y, max_y)
min_x, max_x = -1.2, 1.2
min_y, max_y = -2.0, 1.2
# Create grid of x/y points
num_x_points = 50  # Number of points along the x-axis
num_y_points = 50  # Number of points along the y-axis
######################### GRID OPTIONS

Line = True
#Create distribution of points between point 1 and point 2 for depth calculation
Point1=[548.,381.6,0]
Point2=[743.,395.,0]

#Point1=[1842.47,1429.29,0]
#Point2=[1842.47,961.,0]

#Point1=[2098.,1429.29,0]
#Point2=[2098.,961.,0]

#Point1=[2351.,1429.29,0]
#Point2=[2351.,961.,0]

#Point1=[2041.,1446.0,0]
#Point2=[2088.,1446.,0]
numPoints = 40 # distribution of points between point 1 and point 2 to be vertically integrated based off of vertical direction


# Use the current directory in which ParaView was started
output_directory = os.path.dirname(os.path.abspath(__file__))  # Get the current script directory
if not output_directory:  # Fall back to the current working directory
    output_directory = os.getcwd()

# Generate points between Point1 and Point2
points = np.linspace(Point1, Point2, numPoints)

use_csv=False
#################### IF YOU WANT TO USE A CSV FILE ############################
if use_csv:
    filename = 'ShannonCenter.csv'
    file_path = os.path.join(output_directory, filename)
    # Load the data from the CSV file
    points = np.loadtxt(file_path, delimiter=',')
    numPoints = len(points)

    # Ensure the data has 3 columns (optional, if guaranteed by your file structure)
    if points.shape[1] != 3:
        raise ValueError("The CSV file does not have exactly 3 columns. Add a column fille with zeros (or any number) for your vertical coordinate")


if Line == True:
    #initalize the data_storage for all cases to plot later
    case_data_storage_list = []
    for caseNum in range(len(cases)):
        # Load the OpenFOAM dataset
        filename = os.path.join(output_directory, cases[caseNum], 'case.foam')
        foam_data = OpenFOAMReader(registrationName='case.foam', FileName=filename)
        foam_data.UpdatePipeline()

        # Fetch available time steps
        timesteps = foam_data.TimestepValues
        print("Available timesteps: ", timesteps)

        # Determine desired time step
        if desired_time in ['latest', 'Latest']:
            desired_time = timesteps[-1]  # Grab the latest time step
            print(f"Using latest time step: {desired_time}")
        elif desired_time in timesteps:
            print(f"Using specified time step: {desired_time}")
        else:
            print(f"Desired time {desired_time} not available. Using the default time.")
            desired_time = timesteps[0]  # Fall back to the first available timestep

        # Update the pipeline to the selected time step
        foam_data.UpdatePipeline(time=desired_time)

        #Get domain bounds
        domain_bounds = foam_data.GetDataInformation().GetBounds()  # (xmin, xmax, ymin, ymax, zmin, zmax)
        
        # Initialize a numpy array to store data, now including 8 columns
        data_storage = np.zeros((numPoints, 8))  # Adding two columns for depth_avg_velocity and Froude number

        # Create reusable 'Plot Over Line' and 'Programmable Filter' objects
        plotOverLine = PlotOverLine(registrationName='PlotOverLine', Input=foam_data)
        plotOverLine.SamplingPattern = 'Sample At Cell Boundaries'

        programmableFilter = ProgrammableFilter(registrationName='ProgrammableFilter', Input=plotOverLine)
        programmableFilter.Script = """
        import numpy as np

        g = 9.81

        # Fetch alpha.water, arc_length, and U (velocity vector) arrays
        alpha_water = inputs[0].PointData['alpha.water']
        arc_length = inputs[0].PointData['arc_length']
        U = inputs[0].PointData['U']  # U is the velocity vector array

        # Extract horizontal components based on user input
        U_x = U[:, 0]  # X component of velocity
        U_y = U[:, 1]  # Y component of velocity

        # Initialize sums for alpha_water and velocity components
        total_sum = 0.0
        total_sumX = 0.0
        total_sumY = 0.0

        # Loop through the data to calculate the integrals
        for i in range(1, len(U)):
            if not np.isnan(U_x[i]) and not np.isnan(arc_length[i]) and not np.isnan(arc_length[i-1]):
                differential_length = arc_length[i] - arc_length[i-1]
                total_sum += differential_length*alpha_water[i]
                total_sumX += U_x[i] * differential_length
                total_sumY += U_y[i] * differential_length

        # Compute the depth-averaged velocities if depth is greater than zero
        if total_sum > 0.0:
            depth_avg_U_x = total_sumX / total_sum
            depth_avg_U_y = total_sumY / total_sum
        else:
            depth_avg_U_x = 0.0
            depth_avg_U_y = 0.0

        # Create a depth-averaged velocity vector
        depth_avg_velocity = [depth_avg_U_x, depth_avg_U_y, 0.0]  # Third component is zero as we only have 2D

        # Calculate the velocity magnitude
        velocity_magnitude = np.sqrt(depth_avg_U_x**2 + depth_avg_U_y**2)

        # Calculate the Froude number (Fr = V / sqrt(g * d))
        if total_sum > 0.0:
            froude_number = velocity_magnitude / np.sqrt(g * total_sum)
        else:
            froude_number = 0.0  # Froude number is zero if depth is zero

        # Append the results to FieldData
        output.FieldData.append(total_sum, 'alpha_arc_integral')  # Total depth
        output.FieldData.append(np.array([depth_avg_velocity]), 'depth_avg_velocity')  # Depth-averaged velocity as a vector
        output.FieldData.append(froude_number, 'froude_number')  # Froude number
        """

        # Prepare a list to hold processed results
        results = []

        # Determine bounds based on the specified vertical direction, adding a 1.1x buffer
        if vertical_direction == 0:  # x is vertical
            vertical_min = domain_bounds[0] - 0.05 * (domain_bounds[1] - domain_bounds[0])
            vertical_max = domain_bounds[1] + 0.05 * (domain_bounds[1] - domain_bounds[0])
        elif vertical_direction == 1:  # y is vertical
            vertical_min = domain_bounds[2] - 0.05 * (domain_bounds[3] - domain_bounds[2])
            vertical_max = domain_bounds[3] + 0.05 * (domain_bounds[3] - domain_bounds[2])
        elif vertical_direction == 2:  # z is vertical
            vertical_min = domain_bounds[4] - 0.05 * (domain_bounds[5] - domain_bounds[4])
            vertical_max = domain_bounds[5] + 0.05 * (domain_bounds[5] - domain_bounds[4])

        # Loop through all points
        start_time = time.time()
        for i, point in enumerate(points):
            x, y, z = point  # Unpack the coordinates

            # Calculate the relative position along the line as a percentage
            relative_position = (i / (numPoints - 1)) * 100

            # Set up vertical points with the 1.1x buffer, adjusting based on the chosen vertical direction
            vertical_point1 = [-1.0, -1.0, -1.0]
            vertical_point2 = [1.0, 1.0, 1.0]

            if vertical_direction == 0:  # x is vertical
                vertical_point1[0] = vertical_min
                vertical_point2[0] = vertical_max
                plotOverLine.Point1 = [vertical_point1[0], y, z]
                plotOverLine.Point2 = [vertical_point2[0], y, z]
            elif vertical_direction == 1:  # y is vertical
                vertical_point1[1] = vertical_min
                vertical_point2[1] = vertical_max
                plotOverLine.Point1 = [x, vertical_point1[1], z]
                plotOverLine.Point2 = [x, vertical_point2[1], z]
            elif vertical_direction == 2:  # z is vertical
                vertical_point1[2] = vertical_min
                vertical_point2[2] = vertical_max
                plotOverLine.Point1 = [x, y, vertical_point1[2]]
                plotOverLine.Point2 = [x, y, vertical_point2[2]]

            # Update pipeline only once after setting the points
            plotOverLine.UpdatePipeline()

            # Update the programmable filter to compute alpha_arc_integral
            programmableFilter.UpdatePipeline()

            # Fetch the result from the Programmable Filter
            result = servermanager.Fetch(programmableFilter)

            # Access the computed data in FieldData
            field_data = result.GetFieldData()
            if field_data.GetNumberOfArrays() > 0:
                alpha_arc_integral = field_data.GetArray('alpha_arc_integral').GetValue(0)
                depth_avg_velocity = field_data.GetArray('depth_avg_velocity').GetTuple(0)  # Fetch as a tuple
                froude_number = field_data.GetArray('froude_number').GetValue(0)
                
                # Store depth_avg_velocity as a magnitude
                depth_avg_velocity_mag = np.sqrt(depth_avg_velocity[0]**2 + depth_avg_velocity[1]**2)
                
                results.append([relative_position, x, y, z, alpha_arc_integral, depth_avg_velocity_mag, froude_number])
            else:
                results.append([relative_position, x, y, z, 0, 0, 0])  # Assign 0 if no data is found

            # Time tracking for profiling
            if (len(results) % 10 == 0):  # Print every 10 points for reduced output
                print(f"Processed {len(results)} points in {time.time() - start_time:.2f} seconds")
                start_time = time.time()

        # Convert results to a numpy array for storage
        data_storage = np.array(results)
        case_data_storage_list.append(data_storage)

        # Save data_storage to CSV, updated with new columns
        csv_filename = os.path.join(output_directory, "results.csv")
        try:
            np.savetxt(csv_filename, data_storage, delimiter=",", header="Relative Position along the Line (%),x,y,z,alpha_arc_integral,depth_avg_velocity,Froude_number", comments="")
            print(f"Results saved to {csv_filename}")
        except Exception as e:
            print(f"Error saving CSV file: {e}")
            
    # Generate case labels from the cases list
    case_labels = [case.capitalize() for case in cases]  # Capitalize case names for labels
    num_cases = len(cases)  # Number of cases


    # Plot 1: Depth (alpha_arc_integral)
    plt.figure()
    for i, case_data in enumerate(case_data_storage_list):
        plt.plot(case_data[:, 0], case_data[:, 4], marker='o', label=case_labels[i])
    plt.xlabel('Relative Position along the Line (%)')
    plt.ylabel('Depth')
    plt.title('Depth vs Relative Position along the Line')
    plt.xlim(0, 100)  # Set x-axis limits from 0 to 100
    plt.grid(True)
    plt.legend()
    png_filename = os.path.join(output_directory, "depth_plot.png")
    plt.savefig(png_filename)
    plt.close()

    # Plot 2: Depth-Averaged Velocity
    plt.figure()
    for i, case_data in enumerate(case_data_storage_list):
        plt.plot(case_data[:, 0], case_data[:, 5], marker='o', label=case_labels[i])
    plt.xlabel('Relative Position along the Line (%)')
    plt.ylabel('Depth-Averaged Velocity (m/s)')
    plt.title('Depth-Averaged Velocity vs Relative Position')
    plt.xlim(0, 100)  # Set x-axis limits from 0 to 100
    plt.grid(True)
    plt.legend()
    png_filename = os.path.join(output_directory, "velocity_plot.png")
    plt.savefig(png_filename)
    plt.close()

    # Plot 3: Froude Number
    plt.figure()
    for i, case_data in enumerate(case_data_storage_list):
        plt.plot(case_data[:, 0], case_data[:, 6], marker='o', label=case_labels[i])
    plt.xlabel('Relative Position along the Line (%)')
    plt.ylabel('Froude Number')
    plt.title('Froude Number vs Relative Position')
    plt.xlim(0, 100)  # Set x-axis limits from 0 to 100
    plt.grid(True)
    plt.legend()
    png_filename = os.path.join(output_directory, "froude_number_plot.png")
    plt.savefig(png_filename)
    plt.close()




if grid==True:
    # Get the directory of the current script or working directory
    dir = os.path.dirname(os.path.abspath(__file__))  # Absolute path of the script file
    if not dir:
        dir = os.getcwd()

    # Construct the filename for the OpenFOAM case file
    filename = os.path.join(dir, 'case.foam')

    # Load the OpenFOAM dataset
    foam_data = OpenFOAMReader(registrationName='case.foam', FileName=filename)
    foam_data.UpdatePipeline()

    # Fetch available time steps
    timesteps = foam_data.TimestepValues
    print("Available timesteps: ", timesteps)

    # Check if 'latest' or 'Latest' is requested, otherwise use the specified time
    if desired_time in ['latest', 'Latest']:
        desired_time = timesteps[-1]  # Grab the latest time step
        print(f"Using latest time step: {desired_time}")
    elif desired_time in timesteps:
        print(f"Using specified time step: {desired_time}")
    else:
        print(f"Desired time {desired_time} not available. Using the default time.")
        desired_time = timesteps[0]  # Fall back to the first available timestep

    # Update the pipeline to the selected time step
    foam_data.UpdatePipeline(time=desired_time)

    x_values = np.linspace(min_x, max_x, num_x_points)
    y_values = np.linspace(min_y, max_y, num_y_points)

    # Initialize a numpy array to store data
    data_storage = np.zeros((num_x_points * num_y_points, 6))  # Preallocate for efficiency (including new columns)

    # Create reusable 'Plot Over Line' and 'Programmable Filter' objects
    plotOverLine = PlotOverLine(registrationName='PlotOverLine', Input=foam_data)
    plotOverLine.SamplingPattern = 'Sample At Cell Boundaries'

    programmableFilter = ProgrammableFilter(registrationName='ProgrammableFilter', Input=plotOverLine)
    programmableFilter.Script = """
    import numpy as np

    g = 9.81

    # Fetch alpha.water, arc_length, and U (velocity vector) arrays
    alpha_water = inputs[0].PointData['alpha.water']
    arc_length = inputs[0].PointData['arc_length']
    U = inputs[0].PointData['U']  # U is the velocity vector array

    # Extract horizontal components based on user input
    U_x = U[:, 0]  # X component of velocity
    U_y = U[:, 1]  # Y component of velocity

    # Initialize sums for alpha_water and velocity components
    total_sum = 0.0
    total_sumX = 0.0
    total_sumY = 0.0

    # Loop through the data to calculate the integrals
    for i in range(1, len(alpha_water)):
        if not np.isnan(alpha_water[i]) and not np.isnan(arc_length[i]) and not np.isnan(arc_length[i-1]):
            differential_length = arc_length[i] - arc_length[i-1]
            total_sum += alpha_water[i] * differential_length
            total_sumX += U_x[i] * differential_length * alpha_water[i]
            total_sumY += U_y[i] * differential_length * alpha_water[i]

    # Compute the depth-averaged velocities if depth is greater than zero
    if total_sum > 0.0:
        depth_avg_U_x = total_sumX / total_sum
        depth_avg_U_y = total_sumY / total_sum
    else:
        depth_avg_U_x = 0.0
        depth_avg_U_y = 0.0

    # Create a depth-averaged velocity vector
    depth_avg_velocity = [depth_avg_U_x, depth_avg_U_y, 0.0]  # Third component is zero as we only have 2D

    # Calculate the velocity magnitude
    velocity_magnitude = np.sqrt(depth_avg_U_x**2 + depth_avg_U_y**2)

    # Calculate the Froude number (Fr = V / sqrt(g * d))
    if total_sum > 0.0:
        froude_number = velocity_magnitude / np.sqrt(g * total_sum)
    else:
        froude_number = 0.0  # Froude number is zero if depth is zero

    # Append the results to FieldData
    output.FieldData.append(total_sum, 'alpha_arc_integral')  # Total depth
    output.FieldData.append(np.array([depth_avg_velocity]), 'depth_avg_velocity')  # Depth-averaged velocity as a vector
    output.FieldData.append(froude_number, 'froude_number')  # Froude number
    """

    # Prepare a list to hold processed results
    results = []

    # Loop through all x/y locations
    start_time = time.time()
    for i, x in enumerate(x_values):
        for j, y in enumerate(y_values):
            # Adjust vertical direction based on user input
            vertical_point1 = [-1.0, -1.0, -1.0]
            vertical_point2 = [1.0, 1.0, 1.0]
            vertical_point1[vertical_direction] = -1.0
            vertical_point2[vertical_direction] = 1.0

            # Set non-vertical directions based on user input
            if vertical_direction == 0:  # x is vertical
                plotOverLine.Point1 = [vertical_point1[0], y, x]
                plotOverLine.Point2 = [vertical_point2[0], y, x]
            elif vertical_direction == 1:  # y is vertical
                plotOverLine.Point1 = [x, vertical_point1[1], y]
                plotOverLine.Point2 = [x, vertical_point2[1], y]
            elif vertical_direction == 2:  # z is vertical
                plotOverLine.Point1 = [x, y, vertical_point1[2]]
                plotOverLine.Point2 = [x, y, vertical_point2[2]]

            # Update pipeline only once after setting the points
            plotOverLine.UpdatePipeline()

            # Update the programmable filter to compute alpha_arc_integral
            programmableFilter.UpdatePipeline()

            # Fetch the result from the Programmable Filter
            result = servermanager.Fetch(programmableFilter)

            # Access the computed sum in FieldData
            field_data = result.GetFieldData()
            if field_data.GetNumberOfArrays() > 0:
                alpha_arc_integral = field_data.GetArray('alpha_arc_integral').GetValue(0)
                depth_avg_velocity = field_data.GetArray('depth_avg_velocity')
                froude_number = field_data.GetArray('froude_number').GetValue(0)

                # Append results including new metrics
                print(depth_avg_velocity.GetTuple(0))
                results.append([x, y, 0, alpha_arc_integral, depth_avg_velocity.GetTuple(0)[0], depth_avg_velocity.GetTuple(0)[1], froude_number])
            else:
                results.append([x, y, 0, 0, 0, 0, 0])  # Assign 0 if no data is found

            # Time tracking for profiling
            if (j + 1) % 10 == 0:  # Print every 10 points for reduced output
                print(f"Processed {j + 1} points for x = {x:.2f} in {time.time() - start_time:.2f} seconds")
                start_time = time.time()

    # Convert results to a numpy array for storage
    data_storage = np.array(results)

    # Modify how points are stored and displayed based on the vertical direction
    def apply_vertical_direction(data, vert_dir):
        data[:, [vert_dir, 2]] = data[:, [2, vert_dir]]  # Swap z with vertical

    apply_vertical_direction(data_storage, vertical_direction)

    # Optionally, you can create a Programmable Source to display the stored data
    programmableSource = ProgrammableSource()
    programmableSource.OutputDataSetType = 'vtkPolyData'  # Change to PolyData for points
    programmableSource.Script = """
    import numpy as np
    from vtk import vtkPolyData, vtkPoints, vtkFloatArray

    # Define the data (coordinates, alpha_arc_integral, depth_avg_velocity, and froude_number) you stored earlier
    data = np.array({data})

    # Create points and arrays
    points = vtkPoints()
    depth_array = vtkFloatArray()
    velocity_array = vtkFloatArray()
    froude_array = vtkFloatArray()

    # Set names for arrays
    depth_array.SetName('alpha_arc_integral')
    velocity_array.SetName('depth_avg_velocity')  # Ensure this is treated as a vector
    froude_array.SetName('froude_number')

    # Set the number of components for the velocity array (e.g., 2 for 2D velocity)
    velocity_array.SetNumberOfComponents(2)

    # When filling arrays, ensure the depth_avg_velocity is inserted as a tuple
    for i in range(len(data)):
        points.InsertNextPoint(data[i][:3])  # x, y, z coordinates
        depth_array.InsertNextValue(data[i][3])  # alpha_arc_integral
        velocity_array.InsertNextTuple(data[i][4:6])  # depth_avg_velocity as a tuple
        froude_array.InsertNextValue(data[i][6])  # Froude number

    # Create a PolyData object to store points
    poly_data = vtkPolyData()
    poly_data.SetPoints(points)

    # Add arrays to the points
    poly_data.GetPointData().AddArray(depth_array)
    poly_data.GetPointData().AddArray(velocity_array)
    poly_data.GetPointData().AddArray(froude_array)

    # Set up output
    output.ShallowCopy(poly_data)
    """.format(data=data_storage.tolist())  # Convert to list for string format

    # Update the pipeline and display the stored data
    programmableSource.UpdatePipeline()

    # Show the depth points in ParaView
    renderView1 = GetActiveViewOrCreate('RenderView')
    depthDisplay = Show(programmableSource, renderView1, 'GeometryRepresentation')

    # Optionally color the points by alpha_arc_integral
    ColorBy(depthDisplay, ('POINTS', 'alpha_arc_integral'))

    # Apply Delaunay2D triangulation
    delaunay2D = Delaunay2D(Input=programmableSource)
    delaunay2D.UpdatePipeline()

    # Show the triangulated data
    triangulatedDisplay = Show(delaunay2D, renderView1, 'GeometryRepresentation')
    ColorBy(triangulatedDisplay, ('POINTS', 'alpha_arc_integral'))

    # Reset camera to view all points
    renderView1.ResetCamera()

    # Render all views to see the output
    RenderAllViews()