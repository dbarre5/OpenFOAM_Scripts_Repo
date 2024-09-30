Calculate_AvVel-
	Creates a map of depth averaged velocity, depth, and Fr by resampling your domain on a specified grid.
	You can edit the (GridNumX = 100 GridNumY = 100 GridNumZ = 110) at top of file to change resolution
	You can run this with "pvpython Calculate_AvVel" but you have to load a paraview module first
	Note that this slices your domain at the center depth and ALSO ASSUMES Z is gravity