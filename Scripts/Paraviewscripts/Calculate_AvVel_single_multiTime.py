import os
from paraview.simple import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time

# User inputs vertical direction: 0 for x, 1 for y, 2 for z
vertical_direction = 2



Line = True

plotName = "Line 5"

#Create distribution of points between point 1 and point 2 for depth calculation
Point1=[548.,381.6,0]
Point2=[743.,395.,0]

Point1=[561.73,435.67,0]
Point2=[561.73,293.,0]

Point1=[639.6,435.67,0]
Point2=[639.6,293.,0]

Point1=[716.76,435.67,0]
Point2=[716.76,293.,0]

Point1=[622.25,441.0,0]
Point2=[636.58,441.,0]


numPoints = 40 # distribution of points between point 1 and point 2 to be vertically integrated based off of vertical direction


# Use the current directory in which ParaView was started
output_directory = os.path.dirname(os.path.abspath(__file__))  # Get the current script directory
if not output_directory:  # Fall back to the current working directory
    output_directory = os.getcwd()

# Generate points between Point1 and Point2
points = np.linspace(Point1, Point2, numPoints)

if Line == True:
    #initalize the data_storage for all cases to plot later
    case_data_storage_list = []
    # Load the OpenFOAM dataset
    filename = os.path.join(output_directory, 'case.foam')
    foam_data = OpenFOAMReader(registrationName='case.foam', FileName=filename)
    foam_data.UpdatePipeline()

    # Fetch available time steps
    timesteps = foam_data.TimestepValues
    print("Available timesteps: ", timesteps)
    #loop through timesteps
    for caseNum in range(len(timesteps)):
        desired_time = timesteps[caseNum]

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
        # alpha_water = inputs[0].PointData['alpha.water']
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
                total_sum += differential_length
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
    case_labels = [str(time).capitalize() for time in timesteps]  # Capitalize case names for labels
    num_cases = len(timesteps)  # Number of cases


    # Set up the colormap (viridis) and normalize it based on the number of cases
    num_cases = len(case_data_storage_list)
    colors = cm.viridis(np.linspace(0, 1, num_cases))

    # Calculate 10 evenly spaced tick positions and their corresponding case labels
    num_ticks = 10
    tick_positions = np.linspace(0, num_cases - 1, num_ticks, dtype=int)
    tick_labels = [case_labels[i] for i in tick_positions]

    # Plot 1: Depth (alpha_arc_integral)
    plt.figure()
    for i, case_data in enumerate(case_data_storage_list):
        plt.plot(case_data[:, 0], case_data[:, 4], color=colors[i])
    plt.xlabel('Relative Position along the Line (%)')
    plt.ylabel('Depth')
    plt.title('Depth vs Relative Position along the Line for '+plotName)
    plt.xlim(0, 100)  # Set x-axis limits from 0 to 100
    plt.grid(False)

    # Add color bar with 10 evenly spaced labels
    sm = plt.cm.ScalarMappable(cmap=cm.viridis, norm=plt.Normalize(vmin=0, vmax=num_cases - 1))
    cbar = plt.colorbar(sm, ticks=tick_positions)
    cbar.ax.set_yticklabels(tick_labels)  # Use the 10 interpolated case labels
    cbar.set_label('Time')

    png_filename = os.path.join(output_directory, "depth_plot.png")
    plt.savefig(png_filename, bbox_inches='tight')
    plt.close()

    # Repeat for each plot, adjusting the y-axis label accordingly
    # Plot 2: Depth-Averaged Velocity
    plt.figure()
    for i, case_data in enumerate(case_data_storage_list):
        plt.plot(case_data[:, 0], case_data[:, 5], color=colors[i])
    plt.xlabel('Relative Position along the Line (%)')
    plt.ylabel('Depth-Averaged Velocity (m/s)')
    plt.title('Depth-Averaged Velocity vs Relative Position for '+plotName)
    plt.xlim(0, 100)  # Set x-axis limits from 0 to 100
    plt.grid(False)

    # Color bar with 10 evenly spaced labels
    sm = plt.cm.ScalarMappable(cmap=cm.viridis, norm=plt.Normalize(vmin=0, vmax=num_cases - 1))
    cbar = plt.colorbar(sm, ticks=tick_positions)
    cbar.ax.set_yticklabels(tick_labels)
    cbar.set_label('Time')

    png_filename = os.path.join(output_directory, "velocity_plot.png")
    plt.savefig(png_filename, bbox_inches='tight')
    plt.close()

    # Plot 3: Froude Number
    plt.figure()
    for i, case_data in enumerate(case_data_storage_list):
        plt.plot(case_data[:, 0], case_data[:, 6], color=colors[i])
    plt.xlabel('Relative Position along the Line (%)')
    plt.ylabel('Froude Number')
    plt.title('Froude Number vs Relative Position for '+plotName)
    plt.xlim(0, 100)  # Set x-axis limits from 0 to 100
    plt.grid(False)

    # Color bar with 10 evenly spaced labels
    sm = plt.cm.ScalarMappable(cmap=cm.viridis, norm=plt.Normalize(vmin=0, vmax=num_cases - 1))
    cbar = plt.colorbar(sm, ticks=tick_positions)
    cbar.ax.set_yticklabels(tick_labels)
    cbar.set_label('Time')

    png_filename = os.path.join(output_directory, "froude_number_plot.png")
    plt.savefig(png_filename, bbox_inches='tight')
    plt.close()