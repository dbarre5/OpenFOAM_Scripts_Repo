import os
from paraview.simple import *
import numpy as np
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--desired_time', type=str)
parser.add_argument('--allTimes', action='store_true')
parser.add_argument('--use_station', type=float)
parser.add_argument('--convertToFeet', action='store_true')
parser.add_argument('--sampleUniformly', type=float)
args = parser.parse_args()

# --- Configuration ---
vertical_direction = 2  # 0=x, 1=y, 2=z

pointSet = [
    {"name": "LineA", "type": "2Points", "Point1": [-11.25, 2.79, 6], "Point2": [4.4, 2.79, 6], "numPoints": 400}
]

output_directory = os.path.dirname(os.path.abspath(__file__)) or os.getcwd()
factor = 1.0 / 0.3048 if args.convertToFeet else 1.0
unit   = "ft" if args.convertToFeet else "m"
 
desired_time = None
if args.desired_time:
    try:
        desired_time = float(args.desired_time)
    except ValueError:
        desired_time = args.desired_time  # handle 'latest'
 
# --- Programmable filter ---
# Computes:
#   depth            = integral(alpha * ds)               -> water column thickness
#   depth_avg_speed  = integral(|U| * alpha * ds) / depth -> full 3-component depth-averaged speed
#   froude_number    = depth_avg_speed / sqrt(g * depth)
#   bottom_elevation = lowest vertical location with valid data (bed level)
programmable_script = """
import numpy as np
g = 9.81
vertical_direction = 2
 
alpha_water = inputs[0].PointData['alpha.water']
arc_length  = inputs[0].PointData['arc_length']
U           = inputs[0].PointData['U']
points      = inputs[0].GetPoints()
 
total_sum    = 0.0
total_sumMag = 0.0
vertical_values = []
 
for i in range(1, len(U)):
    ux, uy, uz = U[i, 0], U[i, 1], U[i, 2]
    if not np.isnan(ux) and not np.isnan(arc_length[i]) and not np.isnan(arc_length[i-1]):
        dl   = arc_length[i] - arc_length[i-1]
        a    = alpha_water[i]
        umag = np.sqrt(ux**2 + uy**2 + uz**2)
        total_sum    += dl * a
        total_sumMag += umag * dl * a
        vertical_values.append(points[i][vertical_direction])
 
if total_sum > 0:
    depth_avg_speed = total_sumMag / total_sum
    froude          = depth_avg_speed / np.sqrt(g * total_sum)
else:
    depth_avg_speed = 0.0
    froude          = 0.0
 
# Bottom elevation: lowest valid vertical coordinate (bed surface)
bottom_elev = min(vertical_values) if vertical_values else 0.0
 
output.FieldData.append(total_sum,       'depth')
output.FieldData.append(depth_avg_speed, 'depth_avg_speed')
output.FieldData.append(froude,          'froude_number')
output.FieldData.append(bottom_elev,     'bottom_elevation')
"""
 
 
def make_plots(name, time_step, rows, pos_label, unit, output_directory):
    """
    rows columns: [pos, x, y, z, depth, depth_avg_speed, froude, bottom_elev]
    Produces two stacked subplots sharing the x-axis:
      Top:    WSE and bed profile (filled between)
      Bottom: Froude number with Fr=1 reference line
    """
    pos        = rows[:, 0]
    depth      = rows[:, 4]
    froude     = rows[:, 6]
    bottom     = rows[:, 7]
    wse        = bottom + depth   # water surface elevation
 
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True,
                                   gridspec_kw={'height_ratios': [2, 1]})
 
    # --- Top plot: bed profile + WSE ---
    ax1.fill_between(pos, bottom, wse, alpha=0.35, color='steelblue', label='Water column')
    ax1.fill_between(pos, np.min(bottom) - 0.05 * (np.max(bottom) - np.min(bottom) + 1e-6),
                     bottom, color='sienna', alpha=0.7, label='Bed')
    ax1.plot(pos, wse,    color='steelblue', linewidth=1.5, label='WSE')
    ax1.plot(pos, bottom, color='sienna',    linewidth=1.5, label='Bed elevation')
    ax1.set_ylabel(f"Elevation ({unit})", fontsize=12)
    ax1.set_xlim(0, 100)
    ax1.set_ylim(
        np.min(bottom) - 0.05 * (np.max(bottom) - np.min(bottom) + 1e-6),
        np.max(wse) + 0.80 * (np.max(wse) - np.min(bottom))
    )
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(False)
 
    # --- Bottom plot: Froude number ---
    ax2.plot(pos, froude, color='crimson', linewidth=1.5, label='Froude number')
    ax2.set_ylabel("Froude Number (–)", fontsize=12)
    ax2.set_xlabel(f"{pos_label} ({unit if 'Station' in pos_label else '%'})", fontsize=12)
    ax2.set_xlim(0, 100)
    ax2.set_ylim(0, np.max(froude) * 2.0)
    ax2.legend(fontsize=10, loc='upper right')
    ax2.grid(False)
    ax2.set_ylim(bottom=0)

 
    plt.tight_layout()
    png_path = os.path.join(output_directory, f"{name}_t{time_step}_froude.png")
    plt.savefig(png_path, dpi=150)
    plt.close()
    print(f"Plot saved: {png_path}")
 
 
for ps in pointSet:
    if ps["type"] == "2Points":
        points = np.linspace(ps["Point1"], ps["Point2"], ps["numPoints"])
    elif ps["type"] == "CSV":
        points = np.loadtxt(os.path.join(output_directory, ps["csv_file"]), delimiter=',')
    else:
        print(f"Skipping {ps['name']}: unknown type")
        continue
 
    numPoints = len(points)
    name      = ps["name"]
 
    foam_data = OpenFOAMReader(registrationName='case.foam',
                               FileName=os.path.join(output_directory, 'case.foam'))
    foam_data.UpdatePipeline()
    timesteps = foam_data.TimestepValues
 
    if args.allTimes:
        time_steps_to_process = timesteps
    elif desired_time in ('latest', 'Latest', None):
        time_steps_to_process = [timesteps[-1]]
    elif desired_time in timesteps:
        time_steps_to_process = [desired_time]
    else:
        time_steps_to_process = [timesteps[0]]
 
    bounds = foam_data.GetDataInformation().GetBounds()
    ax_idx = vertical_direction * 2
    vmin   = bounds[ax_idx]     - 0.05 * (bounds[ax_idx+1] - bounds[ax_idx])
    vmax   = bounds[ax_idx + 1] + 0.05 * (bounds[ax_idx+1] - bounds[ax_idx])
 
    plotOverLine = PlotOverLine(registrationName='PlotOverLine', Input=foam_data)
    plotOverLine.SamplingPattern = 'Sample At Cell Boundaries'
    if args.sampleUniformly:
        plotOverLine.SamplingPattern = 'Sample Uniformly'
        plotOverLine.Resolution = int(args.sampleUniformly)
 
    progFilter = ProgrammableFilter(registrationName='ProgrammableFilter', Input=plotOverLine)
    progFilter.Script = programmable_script
 
    for time_step in time_steps_to_process:
        foam_data.UpdatePipeline(time=time_step)
        rows = []
 
        # Cumulative distance along the horizontal sample line
        dists = [0.0]
        for i in range(1, numPoints):
            dists.append(dists[-1] + np.linalg.norm(points[i] - points[i-1]) / factor)
        total_dist = dists[-1]
 
        for i, point in enumerate(points):
            x, y, z = point
 
            if args.use_station is not None:
                pos       = dists[i] + args.use_station
                pos_label = "Station"
            else:
                pos       = (dists[i] / total_dist * 100) if total_dist > 0 else 0.0
                pos_label = "Relative Position"
 
            # Vertical line through this horizontal location
            p1 = [x, y, z]; p2 = [x, y, z]
            p1[vertical_direction] = vmin
            p2[vertical_direction] = vmax
            plotOverLine.Point1 = p1
            plotOverLine.Point2 = p2
            plotOverLine.UpdatePipeline()
            progFilter.UpdatePipeline()
 
            result     = servermanager.Fetch(progFilter)
            field_data = result.GetFieldData()
 
            if field_data.GetNumberOfArrays() > 0:
                depth           = field_data.GetArray('depth').GetValue(0) / factor
                depth_avg_speed = field_data.GetArray('depth_avg_speed').GetValue(0) / factor
                froude          = field_data.GetArray('froude_number').GetValue(0)
                bottom_elev     = field_data.GetArray('bottom_elevation').GetValue(0) / factor
            else:
                depth, depth_avg_speed, froude, bottom_elev = 0.0, 0.0, 0.0, 0.0
 
            rows.append([pos, x, y, z, depth, depth_avg_speed, froude, bottom_elev])
 
            if (i + 1) % 10 == 0:
                print(f"[{name}] t={time_step}  {i+1}/{numPoints} points done")
 
        rows = np.array(rows)
 
        # --- Save CSV ---
        header   = f"{pos_label},x,y,z,Depth ({unit}),Depth_Avg_Speed ({unit}/s),Froude_number,Bottom_Elevation ({unit})"
        csv_path = os.path.join(output_directory, f"{name}_t{time_step}_froude.csv")
        np.savetxt(csv_path, rows, delimiter=",", header=header, comments="")
        print(f"CSV saved: {csv_path}")
 
        # --- Save plot ---
        make_plots(name, time_step, rows, pos_label, unit, output_directory)
 
    Delete(progFilter);   del progFilter
    Delete(plotOverLine); del plotOverLine
    Delete(foam_data);    del foam_data