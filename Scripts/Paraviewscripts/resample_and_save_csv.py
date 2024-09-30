# trace generated using paraview version 5.11.1
import os
from paraview.simple import *

# Define grid dimensions
GridNumX = 100
GridNumY = 100
GridNumZ = 110

# Specify file path
dir = os.path.dirname(__file__)
filename = os.path.join(dir, 'case.foam')

# Create a new 'OpenFOAMReader'
casefoam = OpenFOAMReader(registrationName='case.foam', FileName=filename)
casefoam.MeshRegions = ['internalMesh']
casefoam.CellArrays = []

# Create a new 'Resample To Image'
resampleToImage1 = ResampleToImage(registrationName='ResampleToImage1', Input=casefoam)
resampleToImage1.SamplingDimensions = [GridNumX, GridNumY, GridNumZ]

# Save resampled data to CSV
SaveData('saved_resample_whole_domain.csv', proxy=resampleToImage1, FieldAssociation='Points', AddTime=0)
