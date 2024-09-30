# trace generated using paraview version 5.11.1
#import paraview
#paraview.compatibility.major = 5
#paraview.compatibility.minor = 11
GridNumX = 100
GridNumY = 100
GridNumZ = 110
GridNumXY = GridNumX * GridNumY



import os


dir = os.path.dirname(__file__)
print(dir)
filename = os.path.join(dir, 'case.foam')
#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# create a new 'OpenFOAMReader'
casefoam = OpenFOAMReader(registrationName='case.foam', FileName=filename)
casefoam.MeshRegions = ['internalMesh']
casefoam.CellArrays = ['U', 'alpha.water', 'ddt0(epsilon)', 'ddt0(k)', 'ddt0(rho,U)', 'ddtCorrDdt0(U)', 'epsilon', 'k', 'nut', 'p', 'p_rgh', 'total(p)']

animationScene1 = GetAnimationScene()

# get the time-keeper
timeKeeper1 = GetTimeKeeper()

# update animation scene based on data timesteps
animationScene1.UpdateAnimationUsingDataTimeSteps()

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')

# show data in view
casefoamDisplay = Show(casefoam, renderView1, 'UnstructuredGridRepresentation')

#Find bounds of domain. This is for resampling the image. We create an average z for the slice produced later on
bound = casefoam.GetDataInformation().GetBounds()
z_average = (bound[5]+bound[4]) / 2.0

# get color transfer function/color map for 'p'
pLUT = GetColorTransferFunction('p')

# get opacity transfer function/opacity map for 'p'
pPWF = GetOpacityTransferFunction('p')

# trace defaults for the display properties.
casefoamDisplay.Representation = 'Surface'
casefoamDisplay.ColorArrayName = ['POINTS', 'p']
casefoamDisplay.LookupTable = pLUT
casefoamDisplay.SelectTCoordArray = 'None'
casefoamDisplay.SelectNormalArray = 'None'
casefoamDisplay.SelectTangentArray = 'None'
casefoamDisplay.OSPRayScaleArray = 'p'
casefoamDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
casefoamDisplay.SelectOrientationVectors = 'U'
casefoamDisplay.ScaleFactor = 0.3200000047683716
casefoamDisplay.SelectScaleArray = 'p'
casefoamDisplay.GlyphType = 'Arrow'
casefoamDisplay.GlyphTableIndexArray = 'p'
casefoamDisplay.GaussianRadius = 0.01600000023841858
casefoamDisplay.SetScaleArray = ['POINTS', 'p']
casefoamDisplay.ScaleTransferFunction = 'PiecewiseFunction'
casefoamDisplay.OpacityArray = ['POINTS', 'p']
casefoamDisplay.OpacityTransferFunction = 'PiecewiseFunction'
casefoamDisplay.DataAxesGrid = 'GridAxesRepresentation'
casefoamDisplay.PolarAxes = 'PolarAxesRepresentation'
casefoamDisplay.ScalarOpacityFunction = pPWF
casefoamDisplay.ScalarOpacityUnitDistance = 0.09195467111614546
casefoamDisplay.OpacityArrayName = ['POINTS', 'p']
casefoamDisplay.SelectInputVectors = ['POINTS', 'U']
casefoamDisplay.WriteLog = ''

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
casefoamDisplay.ScaleTransferFunction.Points = [-50.91835021972656, 0.0, 0.5, 0.0, 615.5750122070312, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
casefoamDisplay.OpacityTransferFunction.Points = [-50.91835021972656, 0.0, 0.5, 0.0, 615.5750122070312, 1.0, 0.5, 0.0]

# reset view to fit data
renderView1.ResetCamera(False)

# get the material library
materialLibrary1 = GetMaterialLibrary()

# show color bar/color legend
casefoamDisplay.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# get 2D transfer function for 'p'
pTF2D = GetTransferFunction2D('p')

# create a new 'Calculator'
calculator1 = Calculator(registrationName='Calculator1', Input=casefoam)
calculator1.Function = ''

# Properties modified on calculator1
calculator1.ResultArrayName = 'vel'
calculator1.Function = 'mag(U)'

# show data in view
calculator1Display = Show(calculator1, renderView1, 'UnstructuredGridRepresentation')

# get color transfer function/color map for 'vel'
velLUT = GetColorTransferFunction('vel')

# get opacity transfer function/opacity map for 'vel'
velPWF = GetOpacityTransferFunction('vel')

# trace defaults for the display properties.
calculator1Display.Representation = 'Surface'
calculator1Display.ColorArrayName = ['POINTS', 'vel']
calculator1Display.LookupTable = velLUT
calculator1Display.SelectTCoordArray = 'None'
calculator1Display.SelectNormalArray = 'None'
calculator1Display.SelectTangentArray = 'None'
calculator1Display.OSPRayScaleArray = 'vel'
calculator1Display.OSPRayScaleFunction = 'PiecewiseFunction'
calculator1Display.SelectOrientationVectors = 'U'
calculator1Display.ScaleFactor = 0.3200000047683716
calculator1Display.SelectScaleArray = 'vel'
calculator1Display.GlyphType = 'Arrow'
calculator1Display.GlyphTableIndexArray = 'vel'
calculator1Display.GaussianRadius = 0.01600000023841858
calculator1Display.SetScaleArray = ['POINTS', 'vel']
calculator1Display.ScaleTransferFunction = 'PiecewiseFunction'
calculator1Display.OpacityArray = ['POINTS', 'vel']
calculator1Display.OpacityTransferFunction = 'PiecewiseFunction'
calculator1Display.DataAxesGrid = 'GridAxesRepresentation'
calculator1Display.PolarAxes = 'PolarAxesRepresentation'
calculator1Display.ScalarOpacityFunction = velPWF
calculator1Display.ScalarOpacityUnitDistance = 0.09195467111614546
calculator1Display.OpacityArrayName = ['POINTS', 'vel']
calculator1Display.SelectInputVectors = ['POINTS', 'U']
calculator1Display.WriteLog = ''

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
calculator1Display.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.8567713858683673, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
calculator1Display.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.8567713858683673, 1.0, 0.5, 0.0]

# hide data in view
Hide(casefoam, renderView1)

# show color bar/color legend
calculator1Display.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# get 2D transfer function for 'vel'
velTF2D = GetTransferFunction2D('vel')

# create a new 'Resample To Image'
resampleToImage1 = ResampleToImage(registrationName='ResampleToImage1', Input=calculator1)
resampleToImage1.SamplingBounds = [bound[0], bound[1], bound[2], bound[3], bound[4], bound[5]]
resampleToImage1.SamplingDimensions = [GridNumX, GridNumY, GridNumZ]
# show data in view
resampleToImage1Display = Show(resampleToImage1, renderView1, 'UniformGridRepresentation')

# trace defaults for the display properties.
resampleToImage1Display.Representation = 'Outline'
resampleToImage1Display.ColorArrayName = ['POINTS', 'vel']
resampleToImage1Display.LookupTable = velLUT
resampleToImage1Display.SelectTCoordArray = 'None'
resampleToImage1Display.SelectNormalArray = 'None'
resampleToImage1Display.SelectTangentArray = 'None'
resampleToImage1Display.OSPRayScaleArray = 'vel'
resampleToImage1Display.OSPRayScaleFunction = 'PiecewiseFunction'
resampleToImage1Display.SelectOrientationVectors = 'U'
resampleToImage1Display.ScaleFactor = 0.3199996847683668
resampleToImage1Display.SelectScaleArray = 'vel'
resampleToImage1Display.GlyphType = 'Arrow'
resampleToImage1Display.GlyphTableIndexArray = 'vel'
resampleToImage1Display.GaussianRadius = 0.015999984238418342
resampleToImage1Display.SetScaleArray = ['POINTS', 'vel']
resampleToImage1Display.ScaleTransferFunction = 'PiecewiseFunction'
resampleToImage1Display.OpacityArray = ['POINTS', 'vel']
resampleToImage1Display.OpacityTransferFunction = 'PiecewiseFunction'
resampleToImage1Display.DataAxesGrid = 'GridAxesRepresentation'
resampleToImage1Display.PolarAxes = 'PolarAxesRepresentation'
resampleToImage1Display.ScalarOpacityUnitDistance = 0.04045447443737832
resampleToImage1Display.ScalarOpacityFunction = velPWF
resampleToImage1Display.TransferFunction2D = velTF2D
resampleToImage1Display.OpacityArrayName = ['POINTS', 'vel']
resampleToImage1Display.ColorArray2Name = ['POINTS', 'vel']
resampleToImage1Display.IsosurfaceValues = [0.4259831712855226]
resampleToImage1Display.SliceFunction = 'Plane'
resampleToImage1Display.Slice = 49
resampleToImage1Display.SelectInputVectors = ['POINTS', 'U']
resampleToImage1Display.WriteLog = ''

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
resampleToImage1Display.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.8519663425710452, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
resampleToImage1Display.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.8519663425710452, 1.0, 0.5, 0.0]

# init the 'Plane' selected for 'SliceFunction'
resampleToImage1Display.SliceFunction.Origin = [0.0, -0.3999999761581421, 0.10000000149011609]

# hide data in view
Hide(calculator1, renderView1)

# show color bar/color legend
resampleToImage1Display.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# create a new 'Programmable Filter'
programmableFilter1 = ProgrammableFilter(registrationName='ProgrammableFilter1', Input=resampleToImage1)
programmableFilter1.Script = ''
programmableFilter1.RequestInformationScript = ''
programmableFilter1.RequestUpdateExtentScript = ''
programmableFilter1.PythonPath = ''

# Properties modified on programmableFilter1
script_template = """# Code for \'Script\'.
#Note click "Copy Arrays" option to keep alpha.water to plot surface contour later
import numpy as np
#RequestData (First calc mag(U)-> vel, then ResampleToImage 100x100x100)
input0=inputs[0]
#set up variables
dp = input0.PointData["p"]/9810
dv = input0.PointData["vel"]
da = input0.PointData["alpha.water"]

#Loop through x,y directions
for i in range(0, {GridNumX}):
\tfor j in range(0, {GridNumY}):
\t\t#vertical averaging
\t\tdmax=0
\t\tvsum=0
\t\tasum=0
\t\tfor k in range(0, {GridNumZ}):
\t\t\tid=(k*{GridNumXY})+(j*{GridNumX})+i
\t\t\tvsum=vsum+dv[id]*da[id]
\t\t\tasum=asum+da[id]
\t\t\tif dp[id]>dmax:
\t\t\t\tdmax=dp[id]
\t\t#assign vertical averages throughout depth
\t\tvAv = 0
\t\tif dmax>0:
\t\t\tvAv = vsum/asum
\t\tfor k in range(0, {GridNumZ}): 
\t\t\tid=(k*{GridNumXY})+(j*{GridNumX})+i 
\t\t\tdp[id]=dmax
\t\t\tdv[id]=vAv

output.PointData.append(dp,"depth")
output.PointData.append(dv,"avVel")

"""
# Format the script string with the GridNum
formatted_script = script_template.format(GridNumX=GridNumX,GridNumY=GridNumY,GridNumZ=GridNumZ,GridNumXY=GridNumXY)

# Assign the formatted script to programmableFilter1
programmableFilter1.Script = formatted_script

programmableFilter1.RequestInformationScript = ''
programmableFilter1.RequestUpdateExtentScript = ''
programmableFilter1.CopyArrays = 1
programmableFilter1.PythonPath = ''

# show data in view
programmableFilter1Display = Show(programmableFilter1, renderView1, 'UniformGridRepresentation')

# trace defaults for the display properties.
programmableFilter1Display.Representation = 'Outline'
programmableFilter1Display.ColorArrayName = ['POINTS', 'vel']
programmableFilter1Display.LookupTable = velLUT
programmableFilter1Display.SelectTCoordArray = 'None'
programmableFilter1Display.SelectNormalArray = 'None'
programmableFilter1Display.SelectTangentArray = 'None'
programmableFilter1Display.OSPRayScaleArray = 'vel'
programmableFilter1Display.OSPRayScaleFunction = 'PiecewiseFunction'
programmableFilter1Display.SelectOrientationVectors = 'U'
programmableFilter1Display.ScaleFactor = 0.3199996847683668
programmableFilter1Display.SelectScaleArray = 'vel'
programmableFilter1Display.GlyphType = 'Arrow'
programmableFilter1Display.GlyphTableIndexArray = 'vel'
programmableFilter1Display.GaussianRadius = 0.015999984238418342
programmableFilter1Display.SetScaleArray = ['POINTS', 'vel']
programmableFilter1Display.ScaleTransferFunction = 'PiecewiseFunction'
programmableFilter1Display.OpacityArray = ['POINTS', 'vel']
programmableFilter1Display.OpacityTransferFunction = 'PiecewiseFunction'
programmableFilter1Display.DataAxesGrid = 'GridAxesRepresentation'
programmableFilter1Display.PolarAxes = 'PolarAxesRepresentation'
programmableFilter1Display.ScalarOpacityUnitDistance = 0.04045447443737832
programmableFilter1Display.ScalarOpacityFunction = velPWF
programmableFilter1Display.TransferFunction2D = velTF2D
programmableFilter1Display.OpacityArrayName = ['POINTS', 'vel']
programmableFilter1Display.ColorArray2Name = ['POINTS', 'vel']
programmableFilter1Display.IsosurfaceValues = [0.4259831712855226]
programmableFilter1Display.SliceFunction = 'Plane'
programmableFilter1Display.Slice = 49
programmableFilter1Display.SelectInputVectors = ['POINTS', 'U']
programmableFilter1Display.WriteLog = ''

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
programmableFilter1Display.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.8519663425710452, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
programmableFilter1Display.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 0.8519663425710452, 1.0, 0.5, 0.0]

# init the 'Plane' selected for 'SliceFunction'
programmableFilter1Display.SliceFunction.Origin = [0.0, -0.3999999761581421, 0.10000000149011609]

# hide data in view
Hide(resampleToImage1, renderView1)

# show color bar/color legend
programmableFilter1Display.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# create a new 'Slice'
slice1 = Slice(registrationName='Slice1', Input=programmableFilter1)
slice1.SliceType = 'Plane'
slice1.HyperTreeGridSlicer = 'Plane'
slice1.SliceOffsetValues = [0.0]

# init the 'Plane' selected for 'SliceType'
slice1.SliceType.Origin = [0.0, -0.3999999761581421, z_average]

# init the 'Plane' selected for 'HyperTreeGridSlicer'
slice1.HyperTreeGridSlicer.Origin = [0.0, -0.3999999761581421, z_average]

# Properties modified on slice1.SliceType
slice1.SliceType.Normal = [0.0, 0.0, 1.0]

# show data in view
slice1Display = Show(slice1, renderView1, 'GeometryRepresentation')

# trace defaults for the display properties.
slice1Display.Representation = 'Surface'
slice1Display.ColorArrayName = ['POINTS', 'vel']
slice1Display.LookupTable = velLUT
slice1Display.SelectTCoordArray = 'None'
slice1Display.SelectNormalArray = 'None'
slice1Display.SelectTangentArray = 'None'
slice1Display.OSPRayScaleArray = 'vel'
slice1Display.OSPRayScaleFunction = 'PiecewiseFunction'
slice1Display.SelectOrientationVectors = 'U'
slice1Display.ScaleFactor = 0.31676736472020117
slice1Display.SelectScaleArray = 'vel'
slice1Display.GlyphType = 'Arrow'
slice1Display.GlyphTableIndexArray = 'vel'
slice1Display.GaussianRadius = 0.015838368236010057
slice1Display.SetScaleArray = ['POINTS', 'vel']
slice1Display.ScaleTransferFunction = 'PiecewiseFunction'
slice1Display.OpacityArray = ['POINTS', 'vel']
slice1Display.OpacityTransferFunction = 'PiecewiseFunction'
slice1Display.DataAxesGrid = 'GridAxesRepresentation'
slice1Display.PolarAxes = 'PolarAxesRepresentation'
slice1Display.SelectInputVectors = ['POINTS', 'U']
slice1Display.WriteLog = ''

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
slice1Display.ScaleTransferFunction.Points = [9.608419008251483e-06, 0.0, 0.5, 0.0, 0.6745256147848138, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
slice1Display.OpacityTransferFunction.Points = [9.608419008251483e-06, 0.0, 0.5, 0.0, 0.6745256147848138, 1.0, 0.5, 0.0]

# show color bar/color legend
slice1Display.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# create a new 'Calculator'
calculator2 = Calculator(registrationName='Calculator2', Input=slice1)
calculator2.Function = ''

# Properties modified on calculator2
calculator2.ResultArrayName = 'FroudeN'
calculator2.Function = 'avVel/sqrt(9.81*depth)'

# show data in view
calculator2Display = Show(calculator2, renderView1, 'GeometryRepresentation')

# get color transfer function/color map for 'FroudeN'
froudeNLUT = GetColorTransferFunction('FroudeN')

# trace defaults for the display properties.
calculator2Display.Representation = 'Surface'
calculator2Display.ColorArrayName = ['POINTS', 'FroudeN']
calculator2Display.LookupTable = froudeNLUT
calculator2Display.SelectTCoordArray = 'None'
calculator2Display.SelectNormalArray = 'None'
calculator2Display.SelectTangentArray = 'None'
calculator2Display.OSPRayScaleArray = 'FroudeN'
calculator2Display.OSPRayScaleFunction = 'PiecewiseFunction'
calculator2Display.SelectOrientationVectors = 'U'
calculator2Display.ScaleFactor = 0.31676736472020117
calculator2Display.SelectScaleArray = 'FroudeN'
calculator2Display.GlyphType = 'Arrow'
calculator2Display.GlyphTableIndexArray = 'FroudeN'
calculator2Display.GaussianRadius = 0.015838368236010057
calculator2Display.SetScaleArray = ['POINTS', 'FroudeN']
calculator2Display.ScaleTransferFunction = 'PiecewiseFunction'
calculator2Display.OpacityArray = ['POINTS', 'FroudeN']
calculator2Display.OpacityTransferFunction = 'PiecewiseFunction'
calculator2Display.DataAxesGrid = 'GridAxesRepresentation'
calculator2Display.PolarAxes = 'PolarAxesRepresentation'
calculator2Display.SelectInputVectors = ['POINTS', 'U']
calculator2Display.WriteLog = ''

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
calculator2Display.ScaleTransferFunction.Points = [1.3852828285978656e-05, 0.0, 0.5, 0.0, 3.6130416955098155, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
calculator2Display.OpacityTransferFunction.Points = [1.3852828285978656e-05, 0.0, 0.5, 0.0, 3.6130416955098155, 1.0, 0.5, 0.0]

# hide data in view
Hide(slice1, renderView1)

# show color bar/color legend
calculator2Display.SetScalarBarVisibility(renderView1, True)

# update the view to ensure updated data information
renderView1.Update()

# get opacity transfer function/opacity map for 'FroudeN'
froudeNPWF = GetOpacityTransferFunction('FroudeN')

# get 2D transfer function for 'FroudeN'
froudeNTF2D = GetTransferFunction2D('FroudeN')

# save data
SaveData('C:/Users/RDCHLDDB/Documents/Ubend0.2/withoutWallRefinement/Ubend0.2/Processed_avVel_Dep_Fr.pvd', proxy=calculator2, PointDataArrays=['FroudeN', 'U', 'alpha.water', 'avVel', 'ddt0(epsilon)', 'ddt0(k)', 'ddt0(rho,U)', 'ddtCorrDdt0(U)', 'depth', 'epsilon', 'k', 'nut', 'p', 'p_rgh', 'total(p)', 'vel', 'vtkGhostType', 'vtkValidPointMask'],
    CellDataArrays=['vtkGhostType'],
    FieldDataArrays=['CasePath'])

#================================================================
# addendum: following script captures some of the application
# state to faithfully reproduce the visualization during playback
#================================================================

# get layout
layout1 = GetLayout()

#--------------------------------
# saving layout sizes for layouts

# layout/tab size in pixels
layout1.SetSize(1091, 778)

#-----------------------------------
# saving camera placements for views

# current camera placement for renderView1
renderView1.CameraPosition = [0.0, -0.3999999761581421, 7.837060024857041]
renderView1.CameraFocalPoint = [0.0, -0.3999999761581421, 0.10000000149011612]
renderView1.CameraParallelScale = 2.0024984871487144

#--------------------------------------------
# uncomment the following to render all views
# RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).