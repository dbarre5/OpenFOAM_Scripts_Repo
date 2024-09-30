import os
import re
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import linregress
import numpy as np
from decimal import Decimal
from scipy.optimize import curve_fit

startTime = 0.0
tol = 0.01
# Names of your integrated quantity
fileName = "force.dat"
directory = "postProcessing"
title_font_size_user = 4
# What vector unit are you trying to plot? THIS DOES NOT CHANGE THE ACTUAL UNITS, JUST THE PLOT TITLE
integratedUnit = "N"

#interval of seconds for linear regressions
specified_interval = 20

#Plot exponential decay curve through dataset on last 3 plots? (True/False)
plot_decay = True


# If you want to pecify the startTime of CDF, you won't check for convergence
startTimeCDF = 00.0
if startTimeCDF > 0.0:
    tol = 1e-100
def find_files(directory, prefix):
    files = []
    for root, dirs, filenames in os.walk(directory):
        contains_0 = any(filename.startswith(prefix[:-4] + '_0') for filename in filenames)
        if contains_0:
            files_with_0 = [filename for filename in filenames if filename.startswith(prefix[:-4] + '_0')]
            files.extend(os.path.join(root, filename) for filename in files_with_0)
        else:
            files_without_0 = [filename for filename in filenames if filename.startswith(prefix)]
            files.extend(os.path.join(root, filename) for filename in files_without_0)
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

def aggregate_data(files):
    time_values = set()
    total_force = []
    pressure_force = []
    viscous_force = []

    for filename in files:
        with open(filename, 'r') as file:
            lines = file.readlines()

        time_index = next((i for i, line in enumerate(lines) if 'Time' in line), None)

        if time_index is None:
            print(f"Error: 'Time' not found in the file {filename}.")
            continue
        
        for line in lines[time_index + 1:]:
            parts = line.split()
            time = float(parts[0])
            if time < startTime:
                continue  # Skip data before startTime
            time_values.add(time)
            
            # Extract three vectors from each line
            total_vector = tuple(map(float, parts[1:4]))
            pressure_vector = tuple(map(float, parts[4:7]))
            viscous_vector = tuple(map(float, parts[7:10]))
            
            total_force.append((time, total_vector))
            pressure_force.append((time, pressure_vector))
            viscous_force.append((time, viscous_vector))

    return sorted(time_values), total_force, pressure_force, viscous_force

def split_into_intervals(times, values, interval_seconds):
    interval_points = int(interval_seconds / (times[1] - times[0]))
    intervals = []
    interval_values = []
    current_interval_start = 0
    for i in range(0, len(times), interval_points):
        interval_start = times[i]
        interval_end = times[min(i + interval_points - 1, len(times) - 1)]
        intervals.append((interval_start, interval_end))
        interval_values.append(values[i:i+interval_points])
    return intervals, interval_values

def plot_cumulative_distribution(data, filename_prefix, color, convergence_time, times, pdf):
    plt.figure(figsize=(8, 6))
    # Find the index of the convergence time in the times array
    converged_index = np.searchsorted(times, convergence_time)
    
    # Filter the data to include only values after the convergence point
    data_filtered = data[converged_index:]
    
    # Compute the CDF for the filtered data
    data_sorted = np.sort(data_filtered)
    cdf = np.arange(1, len(data_sorted) + 1) / len(data_sorted)

    plt.plot(data_sorted, cdf, marker='.', linestyle='-', color=color)
    plt.fill_between(data_sorted, cdf, color=color, alpha=0.3)
    
    # Set x-axis limits to the range of the sorted data
    plt.xlim(min(data_sorted), max(data_sorted))
    plt.ylim(0, 1.1)  # Assuming CDF ranges from 0 to 1

    # Add horizontal lines and annotations for percentiles and max value
    percentiles = [50, 75, 90]
    percentile_values = [np.percentile(data_sorted, p) for p in percentiles]
    max_value = np.max(data_sorted)

    for p, val in zip(percentiles, percentile_values):
        cdf_value = np.searchsorted(data_sorted, val) / len(data_sorted)
        line_length = val - np.min(data_sorted)
        plt.plot([np.min(data_sorted), val], [cdf_value, cdf_value], color='r', linestyle='--', linewidth=0.8)
        if line_length < (max(data_sorted) - min(data_sorted)) / 4:  # Check if line length is less than 1/4th of total chart length
            plt.text(val+(max(data_sorted) - min(data_sorted)) / 50, cdf_value, f'{p}th: {val:.2E}', horizontalalignment='left', verticalalignment='bottom', color='r')
        else:
            plt.text(val - line_length / 2, cdf_value, f'{p}th: {val:.2E}', horizontalalignment='center', verticalalignment='bottom', color='r')

    max_cdf_value = len(data_sorted) / len(data_sorted)
    max_line_length = max_value - np.min(data_sorted)
    plt.plot([np.min(data_sorted), max_value], [max_cdf_value, max_cdf_value], color='g', linestyle='--', linewidth=0.8)
    if max_line_length < (max(data_sorted) - min(data_sorted)) / 4:  # Check if line length is less than 1/4th of total chart length
        plt.text(max_value+(max(data_sorted) - min(data_sorted)) / 50, max_cdf_value, f'Max: {max_value:.2E}', horizontalalignment='left', verticalalignment='bottom', color='g')
    else:
        plt.text(max_value - max_line_length / 2, max_cdf_value, f'Max: {max_value:.2E}', horizontalalignment='center', verticalalignment='bottom', color='g')

    plt.xlabel('Value')
    plt.ylabel('Cumulative Probability')
    plt.title(filename_prefix)
    plt.tight_layout()
    pdf.savefig()
    plt.close()
    
def plot_and_save_to_pdf(group_name, time_values, integrated_data, integratedUnit, name, interval_seconds=specified_interval, convergence_tolerance=tol):
    times, values = zip(*integrated_data)
    values_x = [v[0] for v in values]
    values_y = [v[1] for v in values]
    values_z = [v[2] for v in values]

    intervals, interval_values_x = split_into_intervals(times, values_x, interval_seconds)
    _, interval_values_y = split_into_intervals(times, values_y, interval_seconds)
    _, interval_values_z = split_into_intervals(times, values_z, interval_seconds)

    # Colorblind-friendly colors excluding green
    cb_colors = ['#E69F00', '#56B4E9', '#D55E00']  # Yellow, Blue, Red

    # Save plot to PDF
    pdf_filename = f"{name}_force_data_plots.pdf"
    try:
        with PdfPages(pdf_filename) as pdf:
            # Plot with raw data
            fig, ax = plt.subplots()
            ax.plot(times, values_x, label='X component', color=cb_colors[0])
            ax.plot(times, values_y, label='Y component', color=cb_colors[1])
            ax.plot(times, values_z, label='Z component', color=cb_colors[2])

            # Set x-axis limits
            ax.set_xlim(min(times), max(times))

            # Set y-axis limits with whitespace at the top
            y_range = max(max(values_x), max(values_y), max(values_z)) - min(min(values_x), min(values_y), min(values_z))
            ax.set_ylim(min(min(values_x), min(values_y), min(values_z)), max(max(values_x), max(values_y), max(values_z)) + 0.3 * y_range)

            plt.xlabel('Time')
            plt.ylabel(f'Integrated Value ({integratedUnit})')
            plt.title(f'{name} Force vs. Time')
            plt.legend(loc='upper right')  # Legend in the top-right corner
            plt.tight_layout()  # Adjust layout to fit all elements within the figure area
            pdf.savefig()
            plt.close()        
    

            # Store convergence times
            convergence_time_x = startTimeCDF
            convergence_time_y = startTimeCDF
            convergence_time_z = startTimeCDF

            # Flags to indicate if convergence was reached
            convergence_reached_x = False
            convergence_reached_y = False
            convergence_reached_z = False

            # Plot with linear regression lines for different intervals (separate plots for each component)
            for component_values, component_label, color in zip([interval_values_x, interval_values_y, interval_values_z],
                                                                ['X', 'Y', 'Z'], cb_colors):
                fig, ax = plt.subplots()
                ax.plot(times, values_x if component_label == 'X' else (values_y if component_label == 'Y' else values_z),
                        label=f'{component_label} component (Raw Data)', color=color, alpha=0.6, linestyle='', marker='o', markersize=3)
                
                # Set x-axis limits
                ax.set_xlim(min(times), max(times))

                # Calculate y-axis limit with extra space
                component_min_value = min(values_x if component_label == 'X' else (values_y if component_label == 'Y' else values_z))
                component_max_value = max(values_x if component_label == 'X' else (values_y if component_label == 'Y' else values_z))
                component_y_range = component_max_value - component_min_value
                component_buffer = component_y_range * 0.3

                ax.set_ylim(component_min_value - component_buffer, component_max_value + component_buffer)


                # Add vertical black line at startTimeCDF and label it if startTimeCDF > 0.0
                if startTimeCDF > 0.0:
                    ax.axvline(x=startTimeCDF, color='black', linestyle='-', linewidth=1.0)
                    ax.text(startTimeCDF, 0, 'startTimeCDF', ha='right', va='bottom', fontsize=8, color='green', rotation=90)
                    ax.axvspan(startTimeCDF, max(times), color='lightgreen', alpha=0.1)
                # Compute largest slope for the current component
                largest_slope = None
                for vals in component_values:
                    interval_times = np.linspace(intervals[0][0], intervals[0][1], len(vals))  # Use the first interval for reference
                    if not np.isnan(vals).any() and np.isfinite(vals).all():
                        slope, _, _, _, _ = linregress(interval_times, vals)
                        if largest_slope is None or abs(slope) > abs(largest_slope):
                            largest_slope = slope

                if largest_slope is None:
                    print(f"Error: Unable to compute the largest slope for {component_label} component.")
                    continue

                convergence_reached = False
                for i, (interval_start, interval_end) in enumerate(intervals):
                    prev_interval_times = np.linspace(intervals[i-1][0], intervals[i-1][1], len(component_values[i-1]))
                    prev_vals = component_values[i-1]
                    if not np.isnan(prev_vals).any() and np.isfinite(prev_vals).all():
                        prev_slope, _, _, _, _ = linregress(prev_interval_times, prev_vals)
                        slope, _, _, _, _ = linregress(np.linspace(interval_start, interval_end, len(component_values[i])), component_values[i])
                        # Normalize slopes to the largest slope
                        normalized_prev_slope = prev_slope / largest_slope
                        normalized_slope = slope / largest_slope
                        print(f"Interval {i}:")
                        print(f"    Previous Normalized Slope: {normalized_prev_slope}")
                        print(f"    Current Normalized Slope: {normalized_slope}")
                        slope_text = f'{normalized_slope:.2E}'
                        # Adjust position of text
                        y_pos = component_max_value + component_buffer * 0.25 if i % 2 == 0 else component_max_value + component_buffer * 0.35
                        ax.text(interval_start + (interval_end - interval_start) / 2, y_pos,
                                slope_text, ha='center', va='bottom', fontsize=6, color='black')

                        # Check for convergence
                        if abs(normalized_slope) < convergence_tolerance and not convergence_reached:
                            convergence_reached = True
                            convergence_time = interval_start  # Use interval_start for convergence point
                            ax.axvline(x=convergence_time, color='green', linestyle='-', linewidth=1.0)
                            # Store convergence time for the component
                            if component_label == 'X':
                                convergence_time_x = convergence_time
                                convergence_reached_x = True
                            elif component_label == 'Y':
                                convergence_time_y = convergence_time
                                convergence_reached_y = True
                            elif component_label == 'Z':
                                convergence_time_z = convergence_time
                                convergence_reached_z = True
                        else:
                            ax.axvline(x=interval_start, color='grey', linestyle='-', linewidth=0.5)

                    interval_times = np.linspace(interval_start, interval_end, len(component_values[i]))
                    vals = component_values[i]
                    if not np.isnan(vals).any() and np.isfinite(vals).all():
                        slope, intercept, _, _, _ = linregress(interval_times, vals)
                        regression_line_y = [slope * x + intercept for x in interval_times]
                        ax.plot(interval_times, regression_line_y, '--', color='black', linewidth=0.8)
                    
                    # Add vertical lines for intervals
                    ax.axvline(x=interval_start, color='grey', linestyle='-', linewidth=0.5)
                if convergence_reached:
                    print(f"Convergence reached for {component_label} component at time {convergence_time:.2f} seconds.")

                    # Shade the area to the right of the convergence point (green line) if convergence was reached
                    ax.axvspan(convergence_time, max(times), color='lightgreen', alpha=0.3)

                plt.xlabel('Time')
                plt.ylabel(f'Integrated Value ({integratedUnit})')
                plt.title(f'{component_label} component, linear regression, normalized slopes')
                plt.tight_layout()  # Adjust layout to fit all elements within the figure area
                pdf.savefig()
                plt.close()

            # Plot CDF for X, Y, Z raw data values
            
            plot_cumulative_distribution(values_x, f'Cumulative Distribution Function (X component)', cb_colors[0], convergence_time_x, times, pdf)
            plot_cumulative_distribution(values_y, f'Cumulative Distribution Function (Y component)', cb_colors[1], convergence_time_y, times, pdf)
            plot_cumulative_distribution(values_z, f'Cumulative Distribution Function (Z component)', cb_colors[2],convergence_time_z, times, pdf)

            
            
            # Additional plots for each component starting after convergence time with curve fitting
            for values, label, color, convergence_time in zip([values_x, values_y, values_z], ['X', 'Y', 'Z'], cb_colors, 
                                                              [convergence_time_x, convergence_time_y, convergence_time_z]):
                fig, ax = plt.subplots()
                post_convergence_times = [t for t in times if t >= convergence_time]
                post_convergence_values = [v for t, v in zip(times, values) if t >= convergence_time]

                ax.plot(post_convergence_times, post_convergence_values, label=f'{label} component data', color=color)

                # Fit an exponential decay curve to the post-convergence data
                def exp_decay(t, A, k, C):
                    return A * np.exp(-k * (t - convergence_time)) + C

                if len(post_convergence_times) > 1:  # Ensure there is enough data to fit
                    try:
                        if plot_decay:
                            popt, _ = curve_fit(exp_decay, post_convergence_times, post_convergence_values, maxfev=10000)
                            fitted_values = exp_decay(np.array(post_convergence_times), *popt)
                            ax.plot(post_convergence_times, fitted_values, '--', color='black', label='Exponential Decay Fit',linewidth=0.8)

                            # Estimate the value at infinity
                            value_at_infinity = popt[2]  # The offset C represents the value at infinity

                        # Set the y-limit to ensure blank space at the top
                        y_range = max(post_convergence_values) - min(post_convergence_values)
                        ax.set_ylim(min(post_convergence_values), max(post_convergence_values) + y_range * 0.3)

                        # Set the x-limit to include the entire data range without padding to the right
                        ax.set_xlim(min(post_convergence_times), max(post_convergence_times))

                        # Display the annotation in the upper right corner of the plot
                        if plot_decay:
                            ax.text(max(post_convergence_times), max(post_convergence_values) + y_range * 0.2, 
                                    f'Projected Value at Infinity: {value_at_infinity:.3e}   ', ha='right', va='bottom', 
                                    fontsize=8, color='black')               

                    except RuntimeError as e:
                        print(f"Curve fitting failed for {label} component: {e}")               

                # Position legend in top-left corner
                ax.legend(loc='upper left', bbox_to_anchor=(0.02, 0.98), fontsize=8)           
                plt.xlabel('Time')
                plt.ylabel(f'Integrated Value ({integratedUnit})')
                plt.title(f'{label} component for time subsection')
                plt.tight_layout()  # Adjust layout to fit all elements within the figure area
                pdf.savefig()
                plt.close()          
        print(f"Plots saved to {pdf_filename}")

    except Exception as e:
        print(f"Error occurred while saving plots to PDF: {e}")
                        

# Search for the integrated quantities files
integrated_filenames = find_files(directory, fileName)
# Group files by their parent directory
integrated_groups = group_files_by_probe_type(integrated_filenames)

if not integrated_filenames:
    print("Error: " + fileName + " files not found in postProcessing directory.")
    exit()

# Process integrated data files
for group_name, group_files in integrated_groups.items():
    time_values, Total, Pressure, Viscous = aggregate_data(group_files)
    # Plot and save data to PDF
    plot_and_save_to_pdf(group_name, time_values, Total, integratedUnit, name="Total")
    plot_and_save_to_pdf(group_name, time_values, Pressure, integratedUnit, name="Pressure")
    plot_and_save_to_pdf(group_name, time_values, Viscous, integratedUnit, name="Viscous")