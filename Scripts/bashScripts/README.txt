Note that these scripts should be run from your main folder, where you have access to 0, constant, and system


meshCase-
	Takes a file named meshedSurface.obj, extrudes that using extrude2DmeshDict. Then uses extrudeMeshDict to extrude from that mesh


plotResiduals-
	plot residuals from your simulation

jimRun-
	Run script for jim

replaceProbes
	Replaces "probesFileContents" in controlDict with the contents of your probes.csv file. YOu must have a file called probes.csv
	in your main directory and you must run this from main directory.
	Check controlDictOrig for an example of what your file should look like before running this