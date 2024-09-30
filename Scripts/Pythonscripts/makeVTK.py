import os
import vtk
file_name = 'multU'  # or any other file name you're looking for
directory = "postProcessing"


import os
import re
import matplotlib.pyplot as plt
import math
import statistics
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import linregress
from scipy.stats import norm
import numpy as np

startTime = 60.0
linearRegression = True
probabilityDistribution = True
velComponents = False
velMag = True

# Do you want to plot scalar and/or vector?
plotScalar = True
plotVector = True
#Names of your probe datasets
scalar = "multP"
vector = "multU"


# Example usage:
file_name = 'multU'  # or any other file name you're looking for
directory = "postProcessing"


# This is for filtering out data you might not want. Any average pressure/velocity below this value will not be included
tolerance = 1e-5

#PLOTTING
# How many probes do you want on a single plot?
numberOfProbesToGroup = 7
# What vector unit are you trying to plot? THIS DOES NOT CHANGE THE ACTUAL UNITS, JUST THE PLOT TITLE
VelLengthUnit = "Feet per second"
#Units of Scalar unit you want to plot
PUnit = "Pa"
timeUnit = "seconds"

#Do you want to convert data from meters to Feet?
ConvertToFeet = True #False

############################## STATISTICAL PLOTTING ########################
#Do you want to sort values by magnitude
sortByMean = True
############################## STATISTICAL PLOTTING ########################

#Do you want to ESTIMATE the pressure head?
#NOTE THIS DOES NOT TAKE INTO CONSIDERATION DYNAMIC PRESSURE AND IS A VERY ROUGH APPROXIMATION
#THAT SHOULD NOT BE USED FOR ENGINEERING CALCULATIONS
plotPressureHeadApproximation = True



################## DO NOT CHANGE BELOW ############################
lenFac = 1.0
if ConvertToFeet == True:
    lenFac = 3.28
################## DO NOT CHANGE ABOVE ############################
    
#Do you want to 


def find_files(directory, prefix):
    files = []
    for root, dirs, filenames in os.walk(directory):
        for filename in filenames:
            if filename.startswith(prefix):
                files.append(os.path.join(root, filename))
    return files

def group_files_by_probe_type(files):
    grouped_files = {}
    for file in files:
        parts = os.path.normpath(file).split(os.sep)
        probe_type_dir = parts[-3]  # Group by the immediate parent directory name
        if probe_type_dir not in grouped_files:
            grouped_files[probe_type_dir] = []
        grouped_files[probe_type_dir].append(file)
    return grouped_files

def aggregate_data(files, is_vector=True):
    time_values = set()
    data = {}

    for filename in files:
        with open(filename, 'r') as file:
            lines = file.readlines()

        time_index = next((i for i, line in enumerate(lines) if 'Time' in line), None)

        if time_index is None:
            print(f"Error: 'Time' not found in the file {filename}.")
            continue

        for line in lines[time_index + 1:]:
            parts = re.findall(r'[\w.-]+|\(.*?\)', line)
            time = float(parts[0])
            if time < startTime:
                continue  # Skip data before startTime
            time_values.add(time)

            for i, value_str in enumerate(parts[1:]):
                if is_vector:
                    vector = tuple(map(float, value_str.strip('()').split()))
                    value = vector
                else:
                    value = float(value_str)

                probe_id = i + 1
                if probe_id not in data:
                    data[probe_id] = []
                data[probe_id].append((time, value))

    return sorted(time_values), data

def aggregate_scalar_data(files):
    time_values = set()
    scalar_data = {}

    for filename in files:
        with open(filename, 'r') as file:
            lines = file.readlines()

        time_index = next((i for i, line in enumerate(lines) if 'Time' in line), None)

        if time_index is None:
            print(f"Error: 'Time' not found in the file {filename}.")
            continue

        for line in lines[time_index + 1:]:
            data = line.split()
            time = float(data[0])
            if time < startTime:
                continue  # Skip data before startTime
            time_values.add(time)

            for i, value_str in enumerate(data[1:]):
                value = float(value_str)
                probe_id = i + 1
                if probe_id not in scalar_data:
                    scalar_data[probe_id] = []
                scalar_data[probe_id].append((time, value))

    return sorted(time_values), scalar_data

def extract_probe_locations(directory, file_name):
    all_probe_data = {}

    for root, dirs, files in os.walk(directory):
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

def calculate_stats(data):
    stats = {}
    for probe_id, values in data.items():
        if len(values) == 0:
            continue
        if isinstance(values[0][1], tuple):
            magnitudes = [math.sqrt((v[0]*lenFac)**2 + (v[1]*lenFac)**2 + (v[2]*lenFac)**2) for _, v in values]
        else:
            magnitudes = [v for _, v in values]

        min_val = min(magnitudes)
        max_val = max(magnitudes)
        mean_val = statistics.mean(magnitudes)
        q25 = np.percentile(magnitudes, 25)
        q75 = np.percentile(magnitudes, 75)
        iqr = q75 - q25
        max_minus_min = max_val - min_val

        # Linear regression calculation
        xs = [t for t, _ in values]
        slope, _, r_value, _, std_err = linregress(xs, magnitudes)

        stats[probe_id] = {
            'min': min_val,
            'max': max_val,
            'mean': mean_val,
            'q25': q25,
            'q75': q75,
            'q75_Minus_q25': iqr,
            'max_minus_min': max_minus_min,
            'slope_lin_regress': slope,
            'r_squared_lin_regress': r_value**2,
            'std_error_lin_regress': std_err,
        }

    return stats


def save_to_vtk(probe_data, stats, output_file):
    points = vtk.vtkPoints()
    min_values = vtk.vtkFloatArray()
    min_values.SetName("Min")
    max_values = vtk.vtkFloatArray()
    max_values.SetName("Max")
    mean_values = vtk.vtkFloatArray()
    mean_values.SetName("Mean")
    q25_values = vtk.vtkFloatArray()
    q25_values.SetName("Q25")
    q75_values = vtk.vtkFloatArray()
    q75_values.SetName("Q75")
    iqr_values = vtk.vtkFloatArray()
    iqr_values.SetName("Q75_Minus_Q25")
    max_min_diff_values = vtk.vtkFloatArray()
    max_min_diff_values.SetName("Max_Minus_Min")
    slope_values = vtk.vtkFloatArray()
    slope_values.SetName("Slope_lin_regress")
    r_squared_values = vtk.vtkFloatArray()
    r_squared_values.SetName("R_squared_lin_regress")
    std_error_values = vtk.vtkFloatArray()
    std_error_values.SetName("Std_Error_lin_regress")

    for i, (x, y, z) in enumerate(probe_data, start=1):
        points.InsertNextPoint(x, y, z)
        if i in stats:
            min_values.InsertNextValue(stats[i]['min'])
            max_values.InsertNextValue(stats[i]['max'])
            mean_values.InsertNextValue(stats[i]['mean'])
            q25_values.InsertNextValue(stats[i]['q25'])
            q75_values.InsertNextValue(stats[i]['q75'])
            iqr_values.InsertNextValue(stats[i]['q75_Minus_q25'])
            max_min_diff_values.InsertNextValue(stats[i]['max_minus_min'])
            slope_values.InsertNextValue(stats[i]['slope_lin_regress'])
            r_squared_values.InsertNextValue(stats[i]['r_squared_lin_regress'])
            std_error_values.InsertNextValue(stats[i]['std_error_lin_regress'])
        else:
            min_values.InsertNextValue(0.0)
            max_values.InsertNextValue(0.0)
            mean_values.InsertNextValue(0.0)
            q25_values.InsertNextValue(0.0)
            q75_values.InsertNextValue(0.0)
            iqr_values.InsertNextValue(0.0)
            max_min_diff_values.InsertNextValue(0.0)
            slope_values.InsertNextValue(0.0)
            r_squared_values.InsertNextValue(0.0)
            std_error_values.InsertNextValue(0.0)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.GetPointData().AddArray(min_values)
    polydata.GetPointData().AddArray(max_values)
    polydata.GetPointData().AddArray(mean_values)
    polydata.GetPointData().AddArray(q25_values)
    polydata.GetPointData().AddArray(q75_values)
    polydata.GetPointData().AddArray(iqr_values)
    polydata.GetPointData().AddArray(max_min_diff_values)
    polydata.GetPointData().AddArray(slope_values)
    polydata.GetPointData().AddArray(r_squared_values)
    polydata.GetPointData().AddArray(std_error_values)

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(output_file)
    writer.SetInputData(polydata)
    writer.Write()

U_filenames = find_files(directory, vector)
P_filenames = find_files(directory, scalar)
if not U_filenames:
    print(f"Error: '{file_name}' files not found in {directory} directory.")
    exit()

U_groups = group_files_by_probe_type(U_filenames)
P_groups = group_files_by_probe_type(P_filenames)

probe_data_dict = extract_probe_locations(directory, file_name)

# Process U files
for group_name, group_files in U_groups.items():
    time_values, U_data = aggregate_data(group_files, is_vector=True)
    U_stats = calculate_stats(U_data)
    if group_name in probe_data_dict:
        save_to_vtk(probe_data_dict[group_name], U_stats, f"{group_name}_U_probes.vtp")

# Process P files
for group_name, group_files in P_groups.items():
    time_values, P_data = aggregate_scalar_data(group_files)
    P_stats = calculate_stats(P_data)
    if group_name in probe_data_dict:
        save_to_vtk(probe_data_dict[group_name], P_stats, f"{group_name}_P_probes.vtp")