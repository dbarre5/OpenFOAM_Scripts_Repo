import os
import re
import matplotlib.pyplot as plt
import math
import statistics
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import linregress
from scipy.stats import norm
import numpy as np

startTime = 0.0
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

# Search for the "U" and "P" files
U_filenames = find_files("postProcessing", vector)
P_filenames = find_files("postProcessing", scalar)

if not U_filenames:
    print("Error: 'U' files not found in postProcessing directory.")
    exit()

if not P_filenames:
    print("Error: 'P' files not found in postProcessing directory.")
    exit()

# Group files by their parent directory
U_groups = group_files_by_probe_type(U_filenames)
P_groups = group_files_by_probe_type(P_filenames)

def aggregate_data(files):
    time_values = set()
    U_data = {}

    for filename in files:
        with open(filename, 'r') as file:
            lines = file.readlines()
        
        time_index = next((i for i, line in enumerate(lines) if 'Time' in line), None)
        
        if time_index is None:
            print(f"Error: 'Time' not found in the file {filename}.")
            continue

        for line in lines[time_index + 1:]:
            time, *vectors = re.findall(r'[\w.-]+|\(.*?\)', line)
            time = float(time)
            if time < startTime:
                continue  # Skip data before startTime
            time_values.add(time)

            for i, vector_str in enumerate(vectors):
                vector = tuple(map(float, vector_str.strip('()').split()))
                probe_id = i + 1
                if probe_id not in U_data:
                    U_data[probe_id] = []
                U_data[probe_id].append((time, vector))
    
    return sorted(time_values), U_data

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

def aggregate_approximateHead_data(files):
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
            time_values.add(time)

            for i, value_str in enumerate(data[1:]):
                value = float(value_str)/1000.0/9.81
                probe_id = i + 1
                if probe_id not in scalar_data:
                    scalar_data[probe_id] = []
                scalar_data[probe_id].append((time, value))

    return sorted(time_values), scalar_data

def plot_velocity_data(time_values, U_data, filename_prefix):
    num_probes = len(U_data)
    probe_groups = [list(range(i, min(i + numberOfProbesToGroup, num_probes + 1))) for i in range(1, num_probes + 1, numberOfProbesToGroup)]

    pdf_pages = PdfPages(f'{filename_prefix}_velocity_plots_starting_at_{time_values[0]}_s.pdf')

    if velComponents:
        x_velocities = {tuple(probe_group): [] for probe_group in probe_groups}
        y_velocities = {tuple(probe_group): [] for probe_group in probe_groups}
        z_velocities = {tuple(probe_group): [] for probe_group in probe_groups}

        for probe_group in probe_groups:
            for probe_id in probe_group:
                x_vels = [vector[0]*lenFac for _, vector in U_data.get(probe_id, [])]
                y_vels = [vector[1]*lenFac for _, vector in U_data.get(probe_id, [])]
                z_vels = [vector[2]*lenFac for _, vector in U_data.get(probe_id, [])]
                if x_vels:
                    x_velocities[tuple(probe_group)].append(x_vels)
                if y_vels:
                    y_velocities[tuple(probe_group)].append(y_vels)
                if z_vels:
                    z_velocities[tuple(probe_group)].append(z_vels)

        for probe_group, x_vels_group in x_velocities.items():
            plt.figure()
            for i, probe_id in enumerate(probe_group):
                if i < len(x_vels_group):
                    plt.plot(time_values, x_vels_group[i], label=f"Probe {probe_id}")
            plt.xlabel('Time, '+timeUnit)
            plt.ylabel('X Velocity, '+VelLengthUnit)
            plt.title(f'X Velocity vs. Time for Probes {probe_group[0]}-{probe_group[-1]}')
            plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
            plt.grid(True)
            pdf_pages.savefig(bbox_inches='tight')
            plt.close()

        for probe_group, y_vels_group in y_velocities.items():
            plt.figure()
            for i, probe_id in enumerate(probe_group):
                if i < len(y_vels_group):
                    plt.plot(time_values, y_vels_group[i], label=f"Probe {probe_id}")
            plt.xlabel('Time, '+timeUnit)
            plt.ylabel('Y Velocity, '+VelLengthUnit)
            plt.title(f'Y Velocity vs. Time for Probes {probe_group[0]}-{probe_group[-1]}')
            plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
            plt.grid(True)
            pdf_pages.savefig(bbox_inches='tight')
            plt.close()

        for probe_group, z_vels_group in z_velocities.items():
            plt.figure()
            for i, probe_id in enumerate(probe_group):
                if i < len(z_vels_group):
                    plt.plot(time_values, z_vels_group[i], label=f"Probe {probe_id}")
            plt.xlabel('Time, '+timeUnit)
            plt.ylabel('Z Velocity, '+VelLengthUnit)
            plt.title(f'Z Velocity vs. Time for Probes {probe_group[0]}-{probe_group[-1]}')
            plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
            plt.grid(True)
            pdf_pages.savefig(bbox_inches='tight')
            plt.close()

    if velMag:
        mag_velocities = {tuple(probe_group): [] for probe_group in probe_groups}

        for probe_group in probe_groups:
            for probe_id in probe_group:
                mag_vels = [math.sqrt((vector[0]*lenFac)**2 + (vector[1]*lenFac)**2 + (vector[2]*lenFac)**2) for _, vector in U_data.get(probe_id, [])]
                if mag_vels and statistics.mean(mag_vels) > tolerance:
                    mag_velocities[tuple(probe_group)].append(mag_vels)

        for probe_group, mag_vels_group in mag_velocities.items():
            plt.figure()
            for i, probe_id in enumerate(probe_group):
                if i < len(mag_vels_group):
                    plt.plot(time_values, mag_vels_group[i], label=f"Probe {probe_id}")
            plt.xlabel('Time, '+timeUnit)
            plt.ylabel('Velocity Magnitude, '+VelLengthUnit)
            plt.title(f'Velocity Magnitude vs. Time for Probes {probe_group[0]}-{probe_group[-1]}')
            plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
            plt.grid(True)
            pdf_pages.savefig(bbox_inches='tight')
            plt.close()

    pdf_pages.close()

def plot_pressure_data(time_values, P_data, filename_prefix):
    num_probes = len(P_data)
    probe_groups = [list(range(i, min(i + numberOfProbesToGroup, num_probes + 1))) for i in range(1, num_probes + 1, numberOfProbesToGroup)]

    pdf_pages = PdfPages(f'{filename_prefix}_pressure_plots_starting_at_{time_values[0]}_s.pdf')

    for probe_group in probe_groups:
        plt.figure()
        for probe_id in probe_group:
            if probe_id in P_data:
                filtered_data = [(t, p) for t, p in P_data[probe_id] if p > tolerance]
                if filtered_data:
                    filtered_time_values, pressures = zip(*filtered_data)
                    plt.plot(filtered_time_values, pressures, label=f"Probe {probe_id}")
                else:
                    plt.plot([], [], label=f"Probe {probe_id}")
        plt.xlabel('Time, ' + timeUnit)
        plt.ylabel('Pressure Value, ' + PUnit)
        plt.title(f'Pressure Value vs. Time for Probes {probe_group[0]}-{probe_group[-1]}')
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.grid(True)
        pdf_pages.savefig(bbox_inches='tight')
        plt.close()

    pdf_pages.close()
    
def plot_head_data(time_values, head_data, filename_prefix):
    num_probes = len(head_data)
    probe_groups = [list(range(i, min(i + numberOfProbesToGroup, num_probes + 1))) for i in range(1, num_probes + 1, numberOfProbesToGroup)]

    pdf_pages = PdfPages(f'DO_NOT_USE_FOR_REPORTS{filename_prefix}_APPROXIMATE_PRESSURE_HEAD_starting_at_{time_values[0]}_s.pdf')

    for probe_group in probe_groups:
        plt.figure()
        for probe_id in probe_group:
            if probe_id in P_data:
                time_values, pressure_values = zip(*head_data[probe_id])
                plt.plot(time_values, pressure_values, label=f"Probe {probe_id}")
        plt.xlabel('Time, '+timeUnit)
        plt.ylabel('Head Value, meters')
        plt.title('DO NOT USE THIS DATA FOR ENGINEERING CALCULATIONS, ROUGH APPROXIMATION')
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.grid(True)
        pdf_pages.savefig(bbox_inches='tight')
        plt.close()

    pdf_pages.close()

from matplotlib.backends.backend_pdf import PdfPages

from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import numpy as np
import statistics
import math
from scipy.stats import linregress

def plot_statistical_results(time_values, U_data, filename_prefix, sortByMean=False):
    num_probes = len(U_data)
    probe_groups = [list(range(i, min(i + numberOfProbesToGroup, num_probes + 1))) for i in range(1, num_probes + 1, numberOfProbesToGroup)]

    if sortByMean:
        # Calculate mean velocities for each probe group
        mean_velocities_per_group = {}
        for probe_group in probe_groups:
            mean_velocities = []
            for probe_id in probe_group:
                mag_vels = [math.sqrt((vector[0]*lenFac)**2 + (vector[1]*lenFac)**2 + (vector[2]*lenFac)**2) for _, vector in U_data.get(probe_id, [])]
                if mag_vels and statistics.mean(mag_vels) > tolerance:
                    mean_velocities.append(statistics.mean(mag_vels))
            mean_velocities_per_group[tuple(probe_group)] = statistics.mean(mean_velocities)

        # Sort probe groups by mean velocities
        sorted_probe_groups = [group for group, _ in sorted(mean_velocities_per_group.items(), key=lambda x: x[1], reverse=True)]
    else:
        sorted_probe_groups = probe_groups

    pdf_pages = PdfPages(f'{filename_prefix}_box_plots_starting_at_{time_values[0]}_s.pdf')
    
    # Iterate over each probe group
    for group_index, probe_group in enumerate(sorted_probe_groups):
        fig = plt.figure(constrained_layout=True, figsize=(12, 6))
        axs = fig.subplot_mosaic([['Left', 'TopRight'],['Left', 'BottomRight']],
                                  gridspec_kw={'width_ratios':[2, 1]})
        
        probe_ids = []
        velocities = []
        min_values = []
        mean_values = []
        max_values = []

        # Iterate over each probe in the group
        for probe_id in probe_group:
            probe_ids.append(probe_id)
            mag_vels = [math.sqrt((vector[0]*lenFac)**2 + (vector[1]*lenFac)**2 + (vector[2]*lenFac)**2) for _, vector in U_data.get(probe_id, [])]
            
            # Perform linear regression
            slope, intercept, _, _, _ = linregress(time_values, mag_vels)
            
            # Calculate min, max, and mean
            min_val = min(mag_vels) if mag_vels else None
            max_val = max(mag_vels) if mag_vels else None
            mean_val = statistics.mean(mag_vels) if mag_vels else None
            
            min_values.append(min_val)
            max_values.append(max_val)
            mean_values.append(mean_val)
            
            if mean_val is not None and mean_val > tolerance:
                velocities.append(mag_vels)
            else:
                velocities.append([])
        
        # Create box plot without outliers
        box = axs['Left'].boxplot(velocities, vert=False, showfliers=False, patch_artist=True)  
        for patch in box['boxes']:
            patch.set_facecolor('blue')
            patch.set_alpha(0.5) 
        
        axs['Left'].set_yticks(range(1, len(probe_ids) + 1))
        axs['Left'].set_yticklabels(probe_ids)
        axs['Left'].set_xlabel('Velocity Magnitude, '+ VelLengthUnit)
        axs['Left'].set_ylabel('Probe ID')
        axs['Left'].set_title(f'Box Plot of Velocity Magnitude for Probes {probe_ids[0]}-{probe_ids[-1]}')
        axs['Left'].grid(False)

        # Velocity plot on the top right
        axs['TopRight'].set_title('Velocity Plot')
        for probe_id in probe_group:
            mag_vels = [math.sqrt((vector[0]*lenFac)**2 + (vector[1]*lenFac)**2 + (vector[2]*lenFac)**2) for _, vector in U_data.get(probe_id, [])]
            if mag_vels and statistics.mean(mag_vels) > tolerance:
                axs['TopRight'].plot(time_values, mag_vels, label=f"Probe {probe_id}")
        
        # Reverse the order of legend labels
        handles, labels = axs['TopRight'].get_legend_handles_labels()
        axs['TopRight'].legend(reversed(handles), reversed(labels), bbox_to_anchor=(1.05, 1.0), loc='upper left')
        
        axs['TopRight'].set_xlabel('Time, ' + timeUnit)
        axs['TopRight'].set_ylabel('Velocity Magnitude, ' + VelLengthUnit)
        axs['TopRight'].set_title(f'Velocity Magnitude vs. Time for Probes {probe_group[0]}-{probe_group[-1]}')
        axs['TopRight'].grid(True)
        
        # Adjust text size for velocity plot
        for item in ([axs['TopRight'].title, axs['TopRight'].xaxis.label, axs['TopRight'].yaxis.label] +
                     axs['TopRight'].get_xticklabels() + axs['TopRight'].get_yticklabels()):
            item.set_fontsize(8)  # Adjust text size here
        
        # Remove ticks and numbers from the bottom right plot
        axs['BottomRight'].axis('off')
        
        # Add diagnostics in the bottom right, in reverse order
        y_pos = 0.9
        for probe_id, min_val, mean_val, max_val in reversed(list(zip(probe_ids, min_values, mean_values, max_values))):
            axs['BottomRight'].text(
                -0.1, y_pos, f"Probe {probe_id}:", horizontalalignment='left', verticalalignment='center', fontsize=8, fontweight='bold', transform=axs['BottomRight'].transAxes
            )
            axs['BottomRight'].text(
                0.2, y_pos, f"(Min|Mean|Max) = ({min_val:.3f} | {mean_val:.3f} | {max_val:.3f}) {VelLengthUnit}", horizontalalignment='left', verticalalignment='center', fontsize=8, transform=axs['BottomRight'].transAxes
            )
            y_pos -= 0.1
        pdf_pages.savefig(bbox_inches='tight')
        plt.close(fig)

    pdf_pages.close()


def plot_scalar_statistical_results(time_values, P_data, filename_prefix):
    num_probes = len(P_data)
    probe_groups = [list(range(i, min(i + numberOfProbesToGroup, num_probes + 1))) for i in range(1, num_probes + 1, numberOfProbesToGroup)]

    pdf_pages = PdfPages(f'{filename_prefix}_scalar_box_plots_starting_at_{time_values[0]}_s.pdf')
    
    # Prepare data structures to store statistics
    min_values = {}
    max_values = {}
    mean_values = {}

    # Iterate over each probe group
    for group_index, probe_group in enumerate(probe_groups):
        fig = plt.figure(constrained_layout=True, figsize=(12, 6))
        axs = fig.subplot_mosaic([['Left', 'TopRight'],['Left', 'BottomRight']],
                                  gridspec_kw={'width_ratios':[2, 1]})
        
        probe_ids = []
        values = []

        # Iterate over each probe in the group
        for probe_id in probe_group:
            probe_ids.append(probe_id)
            scalar_values = [value for _, value in P_data.get(probe_id, []) if value > tolerance]

            # Calculate min, max, and mean
            min_val = min(scalar_values) if scalar_values else None
            max_val = max(scalar_values) if scalar_values else None
            mean_val = statistics.mean(scalar_values) if scalar_values else None
            min_values[probe_id] = min_val
            max_values[probe_id] = max_val
            mean_values[probe_id] = mean_val
            if scalar_values:
                values.append(scalar_values)
            else:
                values.append([])

        # Create box plot without outliers
        box = axs['Left'].boxplot(values, vert=False, showfliers=False, patch_artist=True)
        for patch in box['boxes']:
            patch.set_facecolor('blue')
            patch.set_alpha(0.5)

        axs['Left'].set_yticks(range(1, len(probe_ids) + 1))
        axs['Left'].set_yticklabels(probe_ids)
        axs['Left'].set_xlabel('Scalar Values')
        axs['Left'].set_ylabel('Probe ID')
        axs['Left'].set_title(f'Box Plot of Scalar Values for Probes {probe_ids[0]}-{probe_ids[-1]}')
        axs['Left'].grid(False)

        # Pressure plot on the top right
        axs['TopRight'].set_title('Pressure Plot')
        for probe_id in probe_group:
            filtered_data = [(t, p) for t, p in P_data.get(probe_id, []) if p > tolerance]
            if filtered_data:
                times, pressures = zip(*filtered_data)
                axs['TopRight'].plot(times, pressures, label=f"Probe {probe_id}")
            else:
                axs['TopRight'].plot([], [], label=f"Probe {probe_id}")

        handles, labels = axs['TopRight'].get_legend_handles_labels()
        axs['TopRight'].legend(reversed(handles), reversed(labels), bbox_to_anchor=(1.05, 1.0), loc='upper left')

        axs['TopRight'].set_xlabel('Time, ' + timeUnit)
        axs['TopRight'].set_ylabel('Pressure Value, ' + PUnit)
        axs['TopRight'].set_title(f'Pressure vs. Time for Probes {probe_group[0]}-{probe_group[-1]}')
        axs['TopRight'].grid(True)

        for item in ([axs['TopRight'].title, axs['TopRight'].xaxis.label, axs['TopRight'].yaxis.label] +
                     axs['TopRight'].get_xticklabels() + axs['TopRight'].get_yticklabels()):
            item.set_fontsize(8)

        # Remove ticks and numbers from the bottom right plot
        axs['BottomRight'].axis('off')

        # Add diagnostics in the bottom right, in reverse order
        y_pos = 0.9
        for probe_id, min_val, mean_val, max_val in reversed(list(zip(probe_ids, min_values.values(), mean_values.values(), max_values.values()))):
            axs['BottomRight'].text(
                -0.1, y_pos, f"Probe {probe_id}:", horizontalalignment='left', verticalalignment='center', fontsize=8, fontweight='bold', transform=axs['BottomRight'].transAxes
            )
            axs['BottomRight'].text(
                0.2, y_pos, f"(Min|Mean|Max) = ({min_val:.3f} | {mean_val:.3f} | {max_val:.3f}) {PUnit}", horizontalalignment='left', verticalalignment='center', fontsize=8, transform=axs['BottomRight'].transAxes
            )
            y_pos -= 0.1

        pdf_pages.savefig(bbox_inches='tight')
        plt.close(fig)

    pdf_pages.close()

def plot_cumulative_distribution(U_data, filename_prefix):
    min_velocities = []
    mean_velocities = []
    max_velocities = []

    for probe_id, data in U_data.items():
        mag_vels = [math.sqrt((vector[0]*lenFac)**2 + (vector[1]*lenFac)**2 + (vector[2]*lenFac)**2) for _, vector in data]
        if mag_vels:
            min_velocity = min(mag_vels)
            mean_velocity = statistics.mean(mag_vels)
            max_velocity = max(mag_vels)
            if mean_velocity > tolerance:
                min_velocities.append(min_velocity)
                mean_velocities.append(mean_velocity)
                max_velocities.append(max_velocity)

    def plot_cdf(data, title, label, color, add_annotations=True):
        data_sorted = np.sort(data)
        cdf = np.arange(1, len(data_sorted) + 1) / len(data_sorted)
        percentiles = [50, 75, 90]
        percentile_values = [np.percentile(data_sorted, p) for p in percentiles]
        max_value = np.max(data_sorted)

        plt.figure(figsize=(10, 6))
        plt.plot(data_sorted, cdf, marker='.', linestyle='-', color=color)
        plt.fill_between(data_sorted, cdf, color=color, alpha=0.3)

        if add_annotations:
            for p, val in zip(percentiles, percentile_values):
                cdf_value = np.searchsorted(data_sorted, val) / len(data_sorted)
                line_length = val - np.min(data_sorted)
                plt.plot([np.min(data_sorted), val], [cdf_value, cdf_value], color='r', linestyle='--', linewidth=0.8)
                plt.text(val - line_length / 2, cdf_value, f'{p}th: {val:.2f} {VelLengthUnit}', horizontalalignment='center', verticalalignment='bottom', color='r')

            max_cdf_value = len(data_sorted) / len(data_sorted)
            max_line_length = max_value - np.min(data_sorted)
            plt.plot([np.min(data_sorted), max_value], [max_cdf_value, max_cdf_value], color='g', linestyle='--', linewidth=0.8)
            plt.text(max_value - max_line_length / 2, max_cdf_value, f'Max: {max_value:.2f} {VelLengthUnit}', horizontalalignment='center', verticalalignment='bottom', color='g')

        plt.title(title)
        plt.xlabel(label + ', ' + VelLengthUnit)
        plt.ylabel('Cumulative Probability')
        plt.legend()
        plt.xlim(np.min(data_sorted), max_value)
        plt.ylim(0, 1.1)

    pdf_filename = f'{filename_prefix}_velocity_cumulative_distribution.pdf'
    pdf_pages = PdfPages(pdf_filename)

    plot_cdf(min_velocities, 'Cumulative Distribution of Min Velocities for all Probes', 'Min Velocity', 'blue')
    pdf_pages.savefig(bbox_inches='tight')
    plt.close()

    plot_cdf(mean_velocities, 'Cumulative Distribution of Mean Velocities for all Probes', 'Mean Velocity', 'green')
    pdf_pages.savefig(bbox_inches='tight')
    plt.close()

    plot_cdf(max_velocities, 'Cumulative Distribution of Max Velocities for all Probes', 'Max Velocity', 'red')
    pdf_pages.savefig(bbox_inches='tight')
    plt.close()

    # Plot overlay of min, mean, and max CDFs without annotations
    def calculate_cdf(data):
        data_sorted = np.sort(data)
        cdf = np.arange(1, len(data_sorted) + 1) / len(data_sorted)
        return data_sorted, cdf

    min_velocities_sorted, min_cdf = calculate_cdf(min_velocities)
    mean_velocities_sorted, mean_cdf = calculate_cdf(mean_velocities)
    max_velocities_sorted, max_cdf = calculate_cdf(max_velocities)

    plt.figure(figsize=(10, 6))
    plt.plot(min_velocities_sorted, min_cdf, linestyle='-', linewidth=1, color='blue', label='Min Velocities')
    plt.plot(mean_velocities_sorted, mean_cdf, linestyle='-', linewidth=1, color='green', label='Mean Velocities')
    plt.plot(max_velocities_sorted, max_cdf, linestyle='-', linewidth=1, color='red', label='Max Velocities')

    plt.fill_between(min_velocities_sorted, min_cdf, color='blue', alpha=0.3)
    plt.fill_between(mean_velocities_sorted, mean_cdf, color='green', alpha=0.3)
    plt.fill_between(max_velocities_sorted, max_cdf, color='red', alpha=0.3)

    plt.title('Cumulative Distribution Function (Min, Mean, Max Velocities)')
    plt.xlabel('Velocity Magnitude, ' + VelLengthUnit)
    plt.ylabel('Cumulative Probability')
    plt.legend()
    plt.xlim(min(np.min(min_velocities_sorted), np.min(mean_velocities_sorted), np.min(max_velocities_sorted)),
             max(np.max(min_velocities_sorted), np.max(mean_velocities_sorted), np.max(max_velocities_sorted)))
    plt.ylim(0, 1.1)

    pdf_pages.savefig(bbox_inches='tight')
    plt.close()

    pdf_pages.close()
    print(f"Cumulative distribution plots saved as {pdf_filename}")
    
    
    
for group_name, group_files in U_groups.items():
    time_values, U_data = aggregate_data(group_files)
    if plotVector:
        plot_velocity_data(time_values, U_data, group_name.replace("\\", "_").replace("/", "_"))
        plot_statistical_results(time_values, U_data, group_name.replace("\\", "_").replace("/", "_"))
        plot_cumulative_distribution(U_data, group_name.replace("\\", "_").replace("/", "_"))



for group_name, group_files in P_groups.items():
    time_values, P_data = aggregate_scalar_data(group_files)
    if plotScalar:
        plot_pressure_data(time_values, P_data, group_name.replace("\\", "_").replace("/", "_"))
        plot_scalar_statistical_results(time_values, P_data, group_name.replace("\\", "_").replace("/", "_"))
    
if plotPressureHeadApproximation == True:   
    for group_name, group_files in P_groups.items():
        time_values, head_data = aggregate_approximateHead_data(group_files)
        plot_head_data(time_values, head_data, group_name.replace("\\", "_").replace("/", "_"))

print("Separated Velocity and Pressure Plots saved as PDF files.")