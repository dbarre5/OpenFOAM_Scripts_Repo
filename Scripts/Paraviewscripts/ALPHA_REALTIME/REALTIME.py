import os
from paraview.simple import *
import numpy as np
import matplotlib.pyplot as plt
import time
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm
import argparse  # Import argparse

# Set up argument parser
parser = argparse.ArgumentParser(description='Process some OpenFOAM data.')
parser.add_argument('--desired_time', type=str, help='Specific time step (e.g., "latest" or a specific time like "10.0")')
parser.add_argument('--allTimes', action='store_true', help='Process all available time steps')
parser.add_argument('--case', type=str, help='The OpenFOAM case directory, does nothing right now')
parser.add_argument('--colormap', type=str, help='Colormap details (palfileName.pal, num_discrete, min_value, max_value)')
# New optional argument --use_station that accepts a float value
parser.add_argument('--use_station', type=float, help='Enable station positioning with a specific starting point (e.g., 0.0)')
# Optional flag to convert to feet
parser.add_argument('--convertToFeet', action='store_true', help='Convert values to feet assuming your simulation is in meters')

# Parse the arguments
args = parser.parse_args()



cases = [""]

# User inputs vertical direction: 0 for x, 1 for y, 2 for z
#### IMPORTANT NOTE YOU MUST GO INTO THE PROGRAMMABLE FILTER AND CHANGE THIS TOO!! SEARCH FOR vertical_direction and change that there!!!
vertical_direction = 2 #### IMPORTANT NOTE YOU MUST GO INTO THE PROGRAMMABLE FILTER AND CHANGE THIS TOO!! SEARCH FOR vertical_direction and change that there!!!

#### IMPORTANT NOTE YOU MUST GO INTO THE PROGRAMMABLE FILTER AND CHANGE THIS TOO!! SEARCH FOR vertical_direction and change that there!!!

# Specify the desired time step (e.g., 'latest' or a specific time like 10.0)
if args.desired_time:
    desired_time = float(args.desired_time)  # Convert the folder name to a float (assuming it's a valid time value)
else:
    desired_time = None  # If no specific time is provided, we'll handle it later

#Create distribution of points between point 1 and point 2 for depth calculation
#Line37
Point1=[0.319334,0.240637,6]
Point2=[0.958215,0.722063,6]
#Line102
#Point1=[-0.0831212,0.391173,6]
#Point2=[-0.249446,1.17356,6]
##Line0.5
#Point1=[-0.401,-0.5,6]
#Point2=[-1.199,-0.5,6]

# Define a list of point sets, each containing the name, type (2Points or CSV), and the relevant points or CSV file
pointSet = [
    {"name": "LineA", "type": "2Points", "Point1": [0.319334,0.240637,6], "Point2": [0.948215,0.712063,6], "numPoints": 40},  # Direct points
    {"name": "LineB", "type": "2Points", "Point1": [-0.0831212,0.391173,6], "Point2": [-0.249446,1.17356,6], "numPoints": 40},  # Direct points
    {"name": "LineC", "type": "2Points", "Point1": [-0.401,-0.5,6], "Point2": [-1.199,-0.5,6], "numPoints": 40},  # Direct points
    #{"name": "PointSet2", "type": "CSV", "csv_file": 'line_points.csv'},
    
    
      # CSV file
]

# Use the current directory in which ParaView was started
output_directory = os.path.dirname(os.path.abspath(__file__))  # Get the current script directory
if not output_directory:  # Fall back to the current working directory
    output_directory = os.getcwd()

# Initialize an empty list to store the points
pointList = []
numPointsList = []

# Process each point set in the pointSet list
for i in range(len(pointSet)):
    point_type = pointSet[i]["type"]  # Get the type of the current point set

    if point_type == "2Points":
        # Get the points and numPoints for the 2Points type
        Point1 = pointSet[i]["Point1"]
        Point2 = pointSet[i]["Point2"]
        numPoints = pointSet[i]["numPoints"]
        numPointsList.append(numPoints)

        # Generate points between Point1 and Point2 using np.linspace
        points = np.linspace(Point1, Point2, numPoints)
        pointList.append(points)
    
    elif point_type == "CSV":
        # Get the CSV filename and load the points
        filename = pointSet[i]["csv_file"]
        file_path = os.path.join(output_directory, filename)
        
        # Load the data from the CSV file
        points = np.loadtxt(file_path, delimiter=',')
        numPoints = len(points)
        numPointsList.append(numPoints)

        # Ensure the data has 3 columns (x, y, z)
        if points.shape[1] != 3:
            raise ValueError(f"The CSV file '{filename}' does not have exactly 3 columns. Add a column filled with zeros (or any number) for your vertical coordinate")
        
        pointList.append(points)
    
    else:
        # If the type is neither '2Points' nor 'CSV', print an error message
        print(f"Error: Point set '{pointSet[i]['name']}' has an invalid 'type' specified: {point_type}")
        continue  # Skip processing this point set

factor = 1.0
YLabel_Unit = 'm'
if args.convertToFeet:
    factor = 3.28
    YLabel_Unit = 'ft'


if args.colormap:
    colormap_info = args.colormap.split(',')
    if len(colormap_info) != 4:
        raise ValueError("The --colormap argument should have exactly 4 components: <pal_file>, <num_discrete>, <min_value>, <max_value>")

    # Extract the parts from the input
    colormapName = colormap_info[0].strip()
    try:
        Pal_num_discrete = int(colormap_info[1].strip())  # Convert to integer
    except ValueError:
        raise ValueError("The number of discrete values should be an integer.")

    try:
        Pal_min_value = float(colormap_info[2].strip())  # Convert to float
        Pal_max_value = float(colormap_info[3].strip())  # Convert to float
    except ValueError:
        raise ValueError("The min and max values should be floating-point numbers.")
    def read_pal_file(pal_file_path):
        colors = []
        with open(pal_file_path, 'r') as f:
            for line in f:
                line = line.strip()  # Remove leading/trailing whitespace
                # Skip empty lines or header lines (e.g., 'PALETTE' or 'AWB Rainbow')
                if not line or line.startswith('PALETTE') or line.startswith('"'):
                    continue
                try:
                    # Split the line and take the last three values as the RGB components
                    parts = line.split()
                    r, g, b = map(int, parts[1:4])  # We ignore the first value (float) and get RGB
                    # Normalize RGB values to [0, 1]
                    colors.append([r / 255.0, g / 255.0, b / 255.0])
                except ValueError:
                    # If the line doesn't have valid RGB values, skip it
                    print(f"Skipping invalid line: {line}")
                    continue
        return np.array(colors)
    pal_file_path = os.path.join(output_directory, colormapName)
    palette = read_pal_file(pal_file_path)
Line=True
if Line == True:
    #initialize the data_storage for all cases to plot later
    for pointListNum in range(len(pointSet)):
        print("Im in the loop")
        points=pointList[pointListNum]
        numPoints = numPointsList[pointListNum]
        case_data_storage_list = []
        for caseNum in range(len(cases)):
            
            pointListName = pointSet[pointListNum]["name"]
            # Load the OpenFOAM dataset
            filename = os.path.join(output_directory, cases[caseNum], 'case.foam')
            foam_data = OpenFOAMReader(registrationName='case.foam', FileName=filename)
            foam_data.UpdatePipeline()

            # Fetch available time steps
            timesteps = foam_data.TimestepValues
            print("Available timesteps: ", timesteps)

            # Determine which time steps to process
            if args.allTimes:  # If --allTimes is specified, loop through all time steps
                time_steps_to_process = timesteps
                print(f"Processing all time steps: {timesteps}")
            elif desired_time in ['latest', 'Latest']:  # If 'latest' is specified
                desired_time = timesteps[-1]  # Grab the latest time step
                time_steps_to_process = [desired_time]
                print(f"Using latest time step: {desired_time}")
            elif desired_time in timesteps:
                time_steps_to_process = [desired_time]  # Process a specific time step
                print(f"Using specified time step: {desired_time}")
            else:
                print(f"Desired time {desired_time} not available. Using the default time.")
                time_steps_to_process = [timesteps[0]]  # Fall back to the first available timestep

            # Process each selected time step
            for time_step in time_steps_to_process:
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
                vertical_direction = 2

                # Fetch alpha.water, arc_length, and U (velocity vector) arrays
                alpha_water = inputs[0].PointData['alpha.water']
                arc_length = inputs[0].PointData['arc_length']
                U = inputs[0].PointData['U']  # U is the velocity vector array
                # Access the points (coordinates) of the dataset
                points = inputs[0].GetPoints()

                # Extract horizontal components based on user input
                U_x = U[:, 0]  # X component of velocity
                U_y = U[:, 1]  # Y component of velocity
                U_z = U[:, 2]  # Z component of velocity


                

                # Extract the vertical location based on the vertical direction
                vertical_location = points[:, vertical_direction]

                # Initialize sums for alpha_water and velocity components
                total_sum = 0.0
                total_sumX = 0.0
                total_sumY = 0.0

                first_valid_arc_length = None
                alpha_contour = 0.5

                # Create a list to store Umag and yValue for each point
                umag_list = []
                vertical_value_list = []
                alpha_list = []
                air_location = []
                VminList = [] # location to store bottom surface

                # Loop through the data to calculate the integrals
                for i in range(1, len(U)):
                    if not np.isnan(U_x[i]) and not np.isnan(arc_length[i]) and not np.isnan(arc_length[i-1]):
                        # Find the first valid arc_length to subtract from (only once)
                        if first_valid_arc_length is None:  # If this is the first valid arc_length
                            # Set the first valid arc_length
                            first_valid_arc_length = arc_length[i]
                        differential_length = arc_length[i] - arc_length[i-1]
                        total_sum += differential_length*alpha_water[i]
                        total_sumX += U_x[i] * differential_length*alpha_water[i]
                        total_sumY += U_y[i] * differential_length*alpha_water[i]
                        Ux = U_x[i] * alpha_water[i]
                        Uy = U_y[i] * alpha_water[i]
                        Umag = np.sqrt((U_x[i]* alpha_water[i])**2 + (U_y[i]* alpha_water[i])**2+(U_z[i]* alpha_water[i])**2)


                        yValue = arc_length[i]

                        # Append Umag and arc_length (yValue) for later use
                        umag_list.append(Umag)
                        vertical_value_list.append(vertical_location[i])

                        
                        alpha_list.append(alpha_water[i])
                        air_location.append(arc_length[i])
                
                # Find the minimum value of vertical_value_list
                min_vertical_value = min(vertical_value_list)
                bottom_value = min_vertical_value-(max(vertical_value_list)-min_vertical_value)*0.01

                # Generate 10 values between the minimum value and 20 (inclusive)
                values_between = np.linspace(min_vertical_value, bottom_value, 10)

                # Append the generated values to VminList
                VminList.extend(values_between)

                # Optionally, append 20.0 directly if you want it as the final value
                VminList.append(bottom_value)


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


                #OUTPUT THE UMAG and Y LIST HERE
                # Append Umag and yValue (arc length) to output for visualization later
                output.FieldData.append(np.array(umag_list), 'velocity_magnitude')  # List of velocity magnitudes (Umag)
                output.FieldData.append(np.array(vertical_value_list), 'arc_length_values')  # List of arc lengths (yValues)
                output.FieldData.append(np.array(alpha_list), 'water_fraction')  # List of waters
                output.FieldData.append(np.array(air_location), 'air_loc')  # List of air location
                output.FieldData.append(np.array(VminList), 'bottom_surface')  # bottom surface location
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
                    if i == 0:
                        cumulative_distance = 0  # First point has zero cumulative distance
                    else:
                        # Calculate Euclidean distance from the previous point to the current point
                        prev_point = points[i - 1]
                        cumulative_distance = np.linalg.norm(np.array(point) - np.array(prev_point)) + cumulative_distance
                total_distance = cumulative_distance/factor
                for i, point in enumerate(points):
                    x, y, z = point  # Unpack the coordinates

                    # Calculate the cumulative distance from the first point to the current point
                    if i == 0:
                        cumulative_distance = 0  # First point has zero cumulative distance
                    else:
                        # Calculate Euclidean distance from the previous point to the current point
                        prev_point = points[i - 1]
                        cumulative_distance = (np.linalg.norm(np.array(point) - np.array(prev_point)))/factor + cumulative_distance
                    
                    if args.use_station is not None:
                        relative_position = cumulative_distance + args.use_station
                        Position_Name = "Absolute"
                        Position_Unit = "m"
                        if args.convertToFeet:
                            Position_Unit = "ft"
                    # Calculate the relative position based on the total distance
                    else:
                        relative_position = (cumulative_distance / total_distance) * 100
                        Position_Name = "Relative"
                        Position_Unit = "%"


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
                        verticalValList=field_data.GetArray('arc_length_values')
                        VelList=field_data.GetArray('velocity_magnitude')
                        alphaList=field_data.GetArray('water_fraction')
                        airLocList=field_data.GetArray('air_loc')
                        bottomList=field_data.GetArray('bottom_surface')
                        LowestBottom = min(bottomList)-0.01 #lowest location in entire mesh, with negative buffer
                        # Initialize empty lists to store values
                        vert_values = []
                        velocities = []
                        alphas=[]
                        airLocs=[]
                        bottomLocs = []

                        # Iterate through the VTK arrays and get values
                        for i in range(verticalValList.GetNumberOfValues()):
                            vert_values.append(verticalValList.GetValue(i)/factor)
                            
                        for i in range(VelList.GetNumberOfValues()):
                            velocities.append(VelList.GetValue(i)/factor)
                        
                        for i in range(alphaList.GetNumberOfValues()):
                            alphas.append(alphaList.GetValue(i))
                        for i in range(airLocList.GetNumberOfValues()):
                            airLocs.append(airLocList.GetValue(i)/factor)
                        for i in range(bottomList.GetNumberOfValues()):
                            bottomLocs.append((bottomList.GetValue(i)/factor))
                            
                        
                        
                        # Store depth_avg_velocity as a magnitude
                        depth_avg_velocity_mag = np.sqrt(depth_avg_velocity[0]**2 + depth_avg_velocity[1]**2)
                        
                        results.append([relative_position, x, y, z, alpha_arc_integral/factor, depth_avg_velocity_mag/factor, froude_number, vert_values, velocities,alphas,airLocs, bottomLocs])
                    else:
                        results.append([relative_position, x, y, z, 0, 0, 0, 0, 0,0,0, 0])  # Assign 0 if no data is found

                    # Time tracking for profiling
                    if (len(results) % 10 == 0):  # Print every 10 points for reduced output
                        print(f"Processed {len(results)} points in {time.time() - start_time:.2f} seconds")
                        start_time = time.time()

                # Convert results to a numpy array for storage
                data_storage = np.array(results)
                case_data_storage_list.append(data_storage)


                sliced_data_storage = data_storage[:, 0:7]
                # Convert the sliced data to a format that can be saved to CSV
                # Flatten any lists (if necessary) or remove them from the data
                flattened_data_storage = []
                for row in sliced_data_storage:
                    # Here we assume that y_values and velocities are in the 7th column (index 6) and need to be excluded
                    flattened_data_storage.append(row)

                # Convert to a numpy array
                flattened_data_storage = np.array(flattened_data_storage)
                # Save data_storage to CSV, updated with new columns
                csv_filename = os.path.join(output_directory, pointListName+"results.csv")
                headerName=Position_Name+" Position along the Line ("+Position_Unit+"),x,y,z,Depth,depth_avg_velocity,Froude_number"
                try:
                    np.savetxt(csv_filename, flattened_data_storage, delimiter=",", header=headerName, comments="")
                    print(f"Results saved to {csv_filename}")
                except Exception as e:
                    print(f"Error saving CSV file: {e}")

                ###################CONTOUR PLOTS
                ###################CONTOUR PLOTS
                ###################CONTOUR PLOTS
                ###################CONTOUR PLOTS
                ###################CONTOUR PLOTS
                ###################CONTOUR PLOTS
                # Flatten the data properly by ensuring the lengths of X, Y, and Z match
                # Flatten the data properly by ensuring the lengths of X, Y, and Z match


                # Iterate over the data to flatten it
                for i, case_data in enumerate(case_data_storage_list):
                    x = case_data[:, 0]  # Relative position or x-coordinate
                    y = case_data[:, 7]  # List of y-values (arc lengths)
                    z = case_data[:, 8]  # List of z-values (velocity magnitudes)
                    alpha = case_data[:, 9]
                    airLoc = case_data[:, 10]

                    bottomSurface = case_data[:, 11]


                    X_flat = []
                    Y_flat = []
                    Z_flat = []
                    X_flat2 = []
                    alpha_flat = []
                    airLoc_flat = []

                    X_bottom_flat = []
                    Y_bottom_flat = [] #flattenned for bottom surface
                    # Repeat the x-values for each y and z value

                    for xi, bottomSurfacei in zip(x, bottomSurface):
                        X_bottom_flat.extend([xi] * len(bottomSurfacei))  # Repeat xi for each value in yi
                        Y_bottom_flat.extend(bottomSurfacei)  # Add all values from yi
                    ScalarValue = np.ones_like(X_bottom_flat)
                        


                    # Repeat the x-values for each y and z value
                    for xi, yi, zi, alphai,airLoci in zip(x, y, z, alpha,airLoc):
                        X_flat.extend([xi] * len(yi))  # Repeat xi for each value in yi
                        Y_flat.extend(yi)  # Add all values from yi
                        Z_flat.extend(zi)  # Add all values from zi
                        X_flat2.extend([xi] * len(airLoci))  # Repeat xi for each value in yi
                        alpha_flat.extend(alphai)  # Add all values from alphai
                        airLoc_flat.extend(airLoci)  # Add all values from alphai

                # Convert to numpy arrays for consistency
                X_flat = np.array(X_flat)
                X_flat2 = np.array(X_flat2)
                Y_flat = np.array(Y_flat)
                Z_flat = np.array(Z_flat)
                alpha_flat = np.array(alpha_flat)
                airLoc_flat = np.array(airLoc_flat)


                # Handle Inf values (e.g., remove corresponding data points)
                # Filter out NaN or Inf values
                valid_indices = np.isfinite(X_flat) & np.isfinite(Y_flat) & np.isfinite(Z_flat) 
                X_flat = X_flat[valid_indices]
                Y_flat = Y_flat[valid_indices]
                Z_flat = Z_flat[valid_indices]
                #alpha_flat = alpha_flat[valid_indices]
                #airLoc_flat = airLoc_flat[valid_indices]


                # Plotting using tricontourf for scattered data
                # Plotting using tricontourf for scattered data
                fig, ax = plt.subplots(figsize=(12, 6))  # Define the figure and axes for plotting
                #plt.figure(figsize=(18, 6))

                # Contour plot for velocity magnitude (Z_flat)
                if args.colormap:
                    contour = plt.tricontourf(X_flat, Y_flat, Z_flat, levels=np.linspace(Pal_min_value, Pal_max_value, Pal_num_discrete), cmap=LinearSegmentedColormap.from_list("custom_cmap", palette))
                else:   
                    contour = plt.tricontourf(X_flat, Y_flat, Z_flat, 200, cmap='viridis')  # Contour plot for velocity magnitude
                # Add black contour lines
                #num_black_contours = 5  # Number of black contour lines
                #contour_levels = np.linspace(np.min(Z_flat), np.max(Z_flat), num_black_contours)  # Define levels for the black contours
        #
                ## Create black contour lines
                #black_contour = plt.tricontour(X_flat, Y_flat, Z_flat, levels=contour_levels, colors='black', linewidths=1)

                cbar = plt.colorbar(contour)
                cbar.set_label('Velocity Magnitude, ('+YLabel_Unit+')/s',fontsize=14)  # Customize the label for the color bar

                # Optionally add contour lines
                if args.colormap:
                    tricontour = plt.tricontour(X_flat, Y_flat, alpha_flat, levels=[0.5], colors='black', linewidths=1)
                else:
                    tricontour = plt.tricontour(X_flat, Y_flat, alpha_flat, levels=[0.5], colors='white', linewidths=1)
                #plt.clabel(tricontour, fmt={0.5: '0.5 alpha_water'}, fontsize=12)
                #tricontour = plt.tricontour(X_bottom_flat, Y_bottom_flat, ScalarValue, levels=[1], colors='black', linewidths=1)

                # Create filled contour plot from 0.5 to 0, and clip any values above 0.5
                # Dark Grey Option colors=["#3A3A3A"]
                if args.colormap:
                    contour = ax.tricontourf(X_flat, Y_flat, alpha_flat, levels=np.linspace(0, 0.5, 10), colors=["white"], vmin=0, vmax=0.5)
                else:
                    contour = ax.tricontourf(X_flat, Y_flat, alpha_flat, levels=np.linspace(0, 0.5, 10), colors=["#440154"], vmin=0, vmax=0.5)
                for i in range(len(X_bottom_flat) - 1):
                        ax.fill_between([X_bottom_flat[i], X_bottom_flat[i+1]], Y_bottom_flat[i], Y_bottom_flat[i+1], color='black')

                ax.plot(X_bottom_flat, Y_bottom_flat, color='black', linestyle='-', linewidth=1)  # This connects all points in the order they appear

                # Add labels and title
                plt.xlabel("Percentage along the line ("+Position_Unit+")",fontsize=14)
                plt.ylabel("Vertical Location ("+YLabel_Unit+")",fontsize=14)
                plt.title(f"Velocity contour plot",fontsize=16)
                #plt.ylim(0.0,0.1)

                # Now, use x_values = case_data[:, 0] for the rainbow-colored line
                x_values = case_data[:, 0]  # This is the data for the rainbow-colored line
                norm = plt.Normalize(0,100)  # Normalize x_values
                cmap = plt.cm.inferno  # Use rainbow colormap

                # Create the plot and colorbar as before
                #sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
                #sm.set_array([])  # We don't need actual data for the colorbar
    #
                ## Define a custom axes for the colorbar using add_axes
                #cbar_ax = fig.add_axes([0.1, 0.1, 0.7, 0.02])  # [left, bottom, width, height]
                #fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")


                # Adjust the plot to leave space for the rainbow line under the x-axis
                plt.subplots_adjust(bottom=0.0)  # Adjust bottom to fit the rainbow line

                # Adjust the plot layout to remove excess whitespace
                plt.subplots_adjust(left=0.1, right=1.0, bottom=0.1)  # Adjust margins to reduce whitespace

                # Create the folder if it doesn't exist
                folder_path = os.path.join(output_directory,  pointListName+"_"+cases[caseNum])
                os.makedirs(folder_path, exist_ok=True)  # This will create the folder if it doesn't exist

                # Save the plot
                png_filename = os.path.join(folder_path, f"{str(time_step)}_Velocity_contour_plot.png")
                plt.savefig(png_filename)
                plt.close()
                # Generate case labels from the cases list

                ###################CONTOUR PLOTS
                ###################CONTOUR PLOTS
                ###################CONTOUR PLOTS
                ###################CONTOUR PLOTS

            case_labels = [case.capitalize() for case in cases]  # Capitalize case names for labels
            num_cases = len(cases)  # Number of cases
            ###################CONTOUR PLOTS

            # Create the folder if it doesn't exist
            folder_path = os.path.join(output_directory, pointListName+"_"+cases[caseNum])
            os.makedirs(folder_path, exist_ok=True)  # This will create the folder if it doesn't exist

            # Plot 1: Depth (alpha_arc_integral)
            plt.figure()
            
            for i, case_data in enumerate(case_data_storage_list):
                print("Im plotting case "+str(i))
                plt.plot(case_data[:, 0], case_data[:, 4], label=case_labels[i])
            plt.xlabel(Position_Name+' Position along the Line ('+Position_Unit+')')
            plt.ylabel('Depth, ('+YLabel_Unit+')')
            plt.title('Depth vs '+Position_Name+' Position along the Line')
            plt.xlim(np.min(case_data[:, 0]), np.max(case_data[:, 0]))  # Set x-axis limits from 0 to 100
            plt.grid(True)
            plt.legend()
            png_filename = os.path.join(folder_path, pointListName+" depth_plot.png")
            plt.savefig(png_filename)
            plt.close()

            # Grab the first element from the list in case_data[n, 11] for all rows. case_data[:,11] is actually a series of arrays that I created so you can create a solid black of 
            # ground in the plot. It's an array where the first value is bottom location, the last value is some value lower than the bottom location.
            # Long story short, if we want the bottom surface, we need to grab this first value for each array.
            first_values = np.array([x[0] for x in case_data[:, 11]])
            #case_data[:, 5] is depth. Subtract from bottom values
            WSE = case_data[:, 4] + first_values
            for i, case_data in enumerate(case_data_storage_list):
                print("Im plotting case "+str(i))
                plt.plot(case_data[:, 0], WSE, label=case_labels[i])
            plt.xlabel(Position_Name+' Position along the Line ('+Position_Unit+')')
            plt.ylabel('WSE, ('+YLabel_Unit+')')
            plt.title('WSE vs '+Position_Name+' Position along the Line')
            plt.xlim(np.min(case_data[:, 0]), np.max(case_data[:, 0]))  # Set x-axis limits from 0 to 100
            plt.grid(True)
            plt.legend()
            png_filename = os.path.join(folder_path, pointListName+" WSE_plot.png")
            plt.savefig(png_filename)
            plt.close()

            # Plot 2: Depth-Averaged Velocity
            plt.figure()
            for i, case_data in enumerate(case_data_storage_list):
                plt.plot(case_data[:, 0], case_data[:, 5], label=case_labels[i])
            plt.xlabel(Position_Name+' Position along the Line ('+Position_Unit+')')
            plt.ylabel('Depth-Averaged Velocity ('+YLabel_Unit+'/s)')
            plt.title('Depth-Averaged Velocity vs '+Position_Name+' Position')
            plt.xlim(np.min(case_data[:, 0]), np.max(case_data[:, 0]))  # Set x-axis limits from 0 to 100
            plt.grid(True)
            plt.legend()
            png_filename = os.path.join(folder_path, pointListName+" velocity_plot.png")
            plt.savefig(png_filename)
            plt.close()

            # Plot 3: Froude Number
            plt.figure()
            for i, case_data in enumerate(case_data_storage_list):
                plt.plot(case_data[:, 0], case_data[:, 6], label=case_labels[i])
            plt.xlabel(Position_Name+' Position along the Line ('+Position_Unit+')')
            plt.ylabel('Froude Number')
            plt.title('Froude Number vs '+Position_Name+' Position')
            plt.xlim(np.min(case_data[:, 0]), np.max(case_data[:, 0]))  # Set x-axis limits from 0 to 100
            plt.grid(True)
            plt.legend()
            png_filename = os.path.join(folder_path, pointListName+" froude_number_plot.png")
            plt.savefig(png_filename)
            plt.close()


            # Create a new column with WSE values (matching the number of rows in flattened_data_storage)
            WSE_column = WSE.reshape(-1, 1)  # Reshaping WSE to make it a 2D column vector

            # Append the WSE column to the existing flattened_data_storage
            updated_data_storage = np.hstack((flattened_data_storage, WSE_column))

            # Add the header with WSE
            headerName += ",WSE"

            # Save the updated data_storage to CSV
            csv_filename = os.path.join(output_directory, pointListName + "results.csv")
            try:
                np.savetxt(csv_filename, updated_data_storage, delimiter=",", header=headerName, comments="")
                print(f"Results saved to {csv_filename}")
            except Exception as e:
                print(f"Error saving CSV file: {e}")