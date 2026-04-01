"""
OpenFOAM Mesh File Generator
============================
Edit the inputs below, then run the script to generate:
  - system/blockMeshDict
  - system/snappyHexMeshDict
  - system/surfaceFeatureExtractDict
  - system/controlDict
  - system/setFieldsDict
  - system/fvSolution
  - 0/U
  - 0/alpha.water
  - 0/p_rgh
  - constant/g
  - constant/hRef
"""

import os

# =============================================================================
# USER INPUTS
# =============================================================================

# Boundary definitions: (stl_name, bc_type, value, level, ks)
# bc_type options:
#   "inlet"   -> variableHeightFlowRateInletVelocity (U), value = flowrate (m3/s)
#              -> variableHeightFlowRate (alpha.water), lowerBound=0.0, upperBound=1.0 (fixed defaults)
#   "outlet"  -> pressureInletOutletVelocity (U), zeroGradient (alpha)
#   "noSlip"  -> noSlip (U), zeroGradient (alpha)
#   "slip"    -> slip (U), zeroGradient (alpha)
# level: refinement level for snappyHexMesh (default 0)
# ks: wall roughness — only relevant for "noSlip" patches:
#   None      -> resolve the wall (low-Re, no wall functions)
#   0 or 0.0  -> smooth wall functions
#   > 0       -> rough wall functions with Ks = ks (m), Cs fixed at 0.5
#   ignored for inlet, outlet, slip
boundaries = [
    ("inlet.stl",  "inlet",  0.0123, 0, None),
    ("bottom.stl", "noSlip", None,   0, 0.0),
    ("outlet.stl", "outlet", None,   0, None),
    ("rock.stl",   "noSlip", None,   0, 0.005),
    ("side.stl",   "slip",   None,   0, None),
    ("top.stl",    "slip",   None,   0, None),
]

# controlDict settings
writeInterval     = 5
purgeWrite        = 0
write_iso_surface = True   # write alpha.water = 0.5 isoSurface
probes = [
    # (x, y, z)
    # (0.5, 0.1, 0.2),
]

# Vertical axis: "x", "y", or "z"
vertical_axis = "z"

# Turbulence model — options: "kOmegaSST", "kEpsilon", "RNGkEpsilon", "realizableKE", "laminar"
turbulence_model = "kOmegaSST"

# Turbulence inlet conditions
turbulence_U         = 1.0    # reference velocity magnitude (m/s)
turbulence_I         = 0.05   # turbulence intensity (e.g. 0.05 = 5%)
turbulence_L         = 0.1    # length scale (m) — used for epsilon

# Refinement regions: (name, type, geometry, (lx, ly, lz))
#   type "box": geometry = upper bound on vertical axis (full XY/XZ/YZ extent)
#   type "stl": geometry = stl filename (must also be in geometry block)
#   (lx, ly, lz): refinement level per axis — min becomes uniform level,
#                 remainder becomes directional levelIncrement
refinement_regions = [
    # ("waterSurface", "box", 1.0,        (2, 2, 2)),
    # ("rock",         "stl", "rock.stl", (3, 3, 3)),
]

# Number of subdomains for parallel decomposition
n_subdomains = 4

# stl_files is derived automatically from boundaries — no need to type them twice
stl_files = [entry[0] for entry in boundaries]

# Location of a point inside the fluid domain (must be set manually)
locationInMesh = (-1, 0, 0.15)

output_dir_system   = os.path.join(os.getcwd(), "system")
output_dir_zero     = os.path.join(os.getcwd(), "0")
output_dir_constant = os.path.join(os.getcwd(), "constant")

# =============================================================================


def write_block_mesh_dict(output_dir="."):
    """
    Scan all STLs in constant/triSurface, compute their combined bounding box,
    add 1% padding, then write blockMeshDict to system/.

    Cell counts:
      - nZ is fixed at 10
      - nX and nY are chosen so the aspect ratio is as close to 1:1:1 as possible
    """

    import struct

    os.makedirs(output_dir, exist_ok=True)

    tri_surface_dir = os.path.join(os.getcwd(), "constant", "triSurface")

    if not os.path.isdir(tri_surface_dir):
        print(f"WARNING: {tri_surface_dir} not found. Skipping blockMeshDict.")
        return

    stl_paths = [
        os.path.join(tri_surface_dir, f)
        for f in os.listdir(tri_surface_dir)
        if f.lower().endswith(".stl")
    ]

    if not stl_paths:
        print(f"WARNING: No STL files found in {tri_surface_dir}. Skipping blockMeshDict.")
        return

    global_min = [float("inf")] * 3
    global_max = [float("-inf")] * 3

    def _parse_stl_ascii(path):
        with open(path, "r", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if line.startswith("vertex"):
                    parts = line.split()
                    yield float(parts[1]), float(parts[2]), float(parts[3])

    def _parse_stl_binary(path):
        with open(path, "rb") as fh:
            fh.read(80)
            n_tri = struct.unpack("<I", fh.read(4))[0]
            for _ in range(n_tri):
                fh.read(12)
                for _ in range(3):
                    x, y, z = struct.unpack("<fff", fh.read(12))
                    yield x, y, z
                fh.read(2)

    for stl_path in stl_paths:
        with open(stl_path, "rb") as fh:
            header = fh.read(256)
        is_ascii = header[:5].lower() == b"solid" and b"facet" in header
        vertices = _parse_stl_ascii(stl_path) if is_ascii else _parse_stl_binary(stl_path)

        for x, y, z in vertices:
            if x < global_min[0]: global_min[0] = x
            if y < global_min[1]: global_min[1] = y
            if z < global_min[2]: global_min[2] = z
            if x > global_max[0]: global_max[0] = x
            if y > global_max[1]: global_max[1] = y
            if z > global_max[2]: global_max[2] = z

        print(f"  Processed: {os.path.basename(stl_path)}")

    padding = [0.01 * (global_max[i] - global_min[i]) for i in range(3)]
    xMin = global_min[0] - padding[0]
    xMax = global_max[0] + padding[0]
    yMin = global_min[1] - padding[1]
    yMax = global_max[1] + padding[1]
    zMin = global_min[2] - padding[2]
    zMax = global_max[2] + padding[2]

    dX = xMax - xMin
    dY = yMax - yMin
    dZ = zMax - zMin

    nZ = 10
    cell_size = dZ / nZ
    nX = max(1, round(dX / cell_size))
    nY = max(1, round(dY / cell_size))

    print(f"  Bounding box (padded): X[{xMin:.4f}, {xMax:.4f}]  "
          f"Y[{yMin:.4f}, {yMax:.4f}]  Z[{zMin:.4f}, {zMax:.4f}]")
    print(f"  Cell counts: nX={nX}  nY={nY}  nZ={nZ}  (target cell size ≈ {cell_size:.4f})")

    content = f"""\
/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2406                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
backgroundMesh
{{
    xMin    {xMin:.6g};
    xMax    {xMax:.6g};
    yMin    {yMin:.6g};
    yMax    {yMax:.6g};
    zMin    {zMin:.6g};
    zMax    {zMax:.6g};
    xCells  {nX};
    yCells  {nY};
    zCells  {nZ};
}}
convertToMeters 1;
vertices
(
    ($:backgroundMesh.xMin $:backgroundMesh.yMin $:backgroundMesh.zMin)
    ($:backgroundMesh.xMax $:backgroundMesh.yMin $:backgroundMesh.zMin)
    ($:backgroundMesh.xMax $:backgroundMesh.yMax $:backgroundMesh.zMin)
    ($:backgroundMesh.xMin $:backgroundMesh.yMax $:backgroundMesh.zMin)
    ($:backgroundMesh.xMin $:backgroundMesh.yMin $:backgroundMesh.zMax)
    ($:backgroundMesh.xMax $:backgroundMesh.yMin $:backgroundMesh.zMax)
    ($:backgroundMesh.xMax $:backgroundMesh.yMax $:backgroundMesh.zMax)
    ($:backgroundMesh.xMin $:backgroundMesh.yMax $:backgroundMesh.zMax)
);
blocks
(
    hex (0 1 2 3 4 5 6 7)
    (
        $:backgroundMesh.xCells
        $:backgroundMesh.yCells
        $:backgroundMesh.zCells
    )
    simpleGrading (1 1 1)
);
edges
(
);
boundary
(
//  Uncomment below to define patches in background mesh
/*
    left
    {{
        type patch;
        faces
        (
            (0 3 7 4)
        );
    }}
    right
    {{
        type patch;
        faces
        (
            (1 5 6 2)
        );
    }}
    bottom
    {{
        type patch;
        faces
        (
            (0 1 2 3)
        );
    }}
    top
    {{
        type patch;
        faces
        (
            (4 7 6 5)
        );
    }}
    back
    {{
        type patch;
        faces
        (
            (0 4 5 1)
        );
    }}
    front
    {{
        type patch;
        faces
        (
            (3 2 6 7)
        );
    }}
*/
);
mergePatchPairs
(
);
// ************************************************************************* //
"""

    path = os.path.join(output_dir, "blockMeshDict")
    with open(path, "w") as f:
        f.write(content)
    print(f"Written: {path}")


def write_snappy_hex_mesh_dict(stl_files, locationInMesh, boundaries, refinement_regions=None, vertical_axis="z", output_dir="."):
    """Generate snappyHexMeshDict from a list of STL filenames."""

    os.makedirs(output_dir, exist_ok=True)

    if refinement_regions is None:
        refinement_regions = []

    # build a lookup: patch_name -> (bc_type, level)
    bc_lookup = {}
    for entry in boundaries:
        stl_name = entry[0]
        bc_type  = entry[1]
        level    = entry[3] if len(entry) > 3 else 0
        bc_lookup[stl_name.replace(".stl", "").replace(".STL", "")] = (bc_type, level)

    # --- geometry block ---
    geometry_entries = ""
    for stl in stl_files:
        patch = stl.replace(".stl", "").replace(".STL", "")
        geometry_entries += f"""
    {patch}
    {{
        type triSurfaceMesh;
        file "{stl}";
    }}
"""

    # --- refinementSurfaces block ---
    refinement_entries = ""
    for stl in stl_files:
        patch = stl.replace(".stl", "").replace(".STL", "")
        bc_type, level = bc_lookup.get(patch, ("noSlip", 0))
        patch_type = "patch" if bc_type in ("inlet", "outlet") else "wall"
        refinement_entries += f"""
        {patch}
        {{
            level ({level} {level});
            patchInfo
            {{
                type {patch_type};
            }}
        }}
"""

    # --- features block ---
    features_entries = ""
    for stl in stl_files:
        emesh = stl.replace(".stl", ".eMesh").replace(".STL", ".eMesh")
        features_entries += f"""
        {{
            file "{emesh}";
            level 0;
        }}
"""

    # --- refinement regions geometry entries ---
    BIG  =  1e6
    axis = vertical_axis.lower().strip()

    for name, rtype, geom, levels in refinement_regions:
        if rtype == "box":
            upper = geom
            if axis == "x":
                box_min = f"({-BIG:.0f} {-BIG:.0f} {-BIG:.0f})"
                box_max = f"({upper} {BIG:.0f} {BIG:.0f})"
            elif axis == "y":
                box_min = f"({-BIG:.0f} {-BIG:.0f} {-BIG:.0f})"
                box_max = f"({BIG:.0f} {upper} {BIG:.0f})"
            else:
                box_min = f"({-BIG:.0f} {-BIG:.0f} {-BIG:.0f})"
                box_max = f"({BIG:.0f} {BIG:.0f} {upper})"
            geometry_entries += f"""
    {name}
    {{
        type searchableBox;
        min {box_min};
        max {box_max};
    }}
"""
        elif rtype == "stl":
            patch = geom.replace(".stl", "").replace(".STL", "")
            if geom not in stl_files:
                geometry_entries += f"""
    {patch}
    {{
        type triSurfaceMesh;
        file "{geom}";
    }}
"""

    # --- refinementRegions block ---
    refinement_regions_entries = ""
    for name, rtype, geom, levels in refinement_regions:
        lx, ly, lz = levels
        uniform_level = min(lx, ly, lz)
        dx = lx - uniform_level
        dy = ly - uniform_level
        dz = lz - uniform_level

        entry = f"""
        {name}
        {{
            mode inside;
            levels (( 1e15 {uniform_level} ));
"""
        if dx > 0 or dy > 0 or dz > 0:
            entry += f"            levelIncrement ({uniform_level} {uniform_level} ({dx} {dy} {dz}));\n"
        entry += "        }\n"
        refinement_regions_entries += entry

    loc = f"({locationInMesh[0]} {locationInMesh[1]} {locationInMesh[2]})"

    content = f"""\
/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2406                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      snappyHexMeshDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

castellatedMesh true;
snap            true;
addLayers       false;

geometry
{{{geometry_entries}}};

castellatedMeshControls
{{
    maxLocalCells       100000;
    maxGlobalCells      2000000;
    minRefinementCells  0;
    maxLoadUnbalance    0.10;
    nCellsBetweenLevels 1;

    features
    ({features_entries});

    refinementSurfaces
    {{{refinement_entries}}}

    resolveFeatureAngle 30;
    planarAngle         30;

    refinementRegions
    {{{refinement_regions_entries}}}

    locationInMesh {loc};

    allowFreeStandingZoneFaces true;
}}

snapControls
{{
    nSmoothPatch        3;
    tolerance           2.0;
    nSolveIter          30;
    nRelaxIter          5;

    nFeatureSnapIter    10;
    implicitFeatureSnap false;
    explicitFeatureSnap true;
    multiRegionFeatureSnap false;
}}

addLayersControls
{{
    relativeSizes       true;
    expansionRatio      1.0;
    finalLayerThickness 0.3;
    minThickness        0.25;

    layers
    {{
    }}

    nGrow 0;

    featureAngle            130;
    maxFaceThicknessRatio   0.5;

    nSmoothSurfaceNormals   1;
    nSmoothThickness        10;

    minMedialAxisAngle      90;
    maxThicknessToMedialRatio 0.3;
    nSmoothNormals          3;

    slipFeatureAngle        30;
    nRelaxIter              5;
    nBufferCellsNoExtrude   0;
    nLayerIter              50;
    nRelaxedIter            20;
}}

meshQualityControls
{{
    maxNonOrtho         65;
    maxBoundarySkewness 20;
    maxInternalSkewness 4;
    maxConcave          80;
    minFlatness         0.5;
    minVol              1e-13;
    minTetQuality       1e-30;
    minArea             -1;
    minTwist            0.02;
    minDeterminant      0.001;
    minFaceWeight       0.02;
    minVolRatio         0.01;
    minTriangleTwist    -1;

    relaxed
    {{
        maxNonOrtho 75;
    }}

    nSmoothScale    4;
    errorReduction  0.75;
}}

mergeTolerance 1e-6;

// ************************************************************************* //
"""

    path = os.path.join(output_dir, "snappyHexMeshDict")
    with open(path, "w") as f:
        f.write(content)
    print(f"Written: {path}")


def write_surface_feature_extract_dict(stl_files, output_dir="."):
    """Generate surfaceFeatureExtractDict from a list of STL filenames."""

    os.makedirs(output_dir, exist_ok=True)

    surface_entries = ""
    for stl in stl_files:
        surface_entries += f"""
    {stl}
    {{
        extractionMethod    extractFromSurface;

        extractFromSurfaceCoeffs
        {{
            includedAngle   150;
        }}

        writeObj yes;
    }}
"""

    content = f"""\
/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2406                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      surfaceFeatureExtractDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
{surface_entries}
// ************************************************************************* //
"""

    path = os.path.join(output_dir, "surfaceFeatureExtractDict")
    with open(path, "w") as f:
        f.write(content)
    print(f"Written: {path}")


def write_U(boundaries, output_dir="."):
    """Generate 0/U for interFoam."""

    os.makedirs(output_dir, exist_ok=True)

    patch_entries = ""
    for stl_name, bc_type, value, *_ in boundaries:
        patch = stl_name.replace(".stl", "").replace(".STL", "")

        if bc_type == "inlet":
            patch_entries += f"""
    {patch}
    {{
        type            variableHeightFlowRateInletVelocity;
        flowRate        {value};  //(m3/s)
        alpha           alpha.water;
        value           uniform (0 0 0);
    }}
"""
        elif bc_type == "outlet":
            patch_entries += f"""
    {patch}
    {{
        type            pressureInletOutletVelocity;
        value           uniform (0 0 0);
    }}
"""
        elif bc_type == "noSlip":
            patch_entries += f"""
    {patch}
    {{
        type            noSlip;
    }}
"""
        elif bc_type == "slip":
            patch_entries += f"""
    {patch}
    {{
        type            slip;
    }}
"""
        else:
            raise ValueError(f"Unknown bc_type '{bc_type}' for patch '{patch}'. "
                             f"Valid options: inlet, outlet, noSlip, slip.")

    content = f"""\
/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2406                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       volVectorField;
    object      U;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 1 -1 0 0 0 0];

internalField   uniform (0 0 0);

boundaryField
{{{patch_entries}}}

// ************************************************************************* //
"""

    path = os.path.join(output_dir, "U")
    with open(path, "w") as f:
        f.write(content)
    print(f"Written: {path}")


def write_alpha_water(boundaries, output_dir="."):
    """Generate 0/alpha.water for interFoam."""

    os.makedirs(output_dir, exist_ok=True)

    patch_entries = ""
    for stl_name, bc_type, *_ in boundaries:
        patch = stl_name.replace(".stl", "").replace(".STL", "")

        if bc_type == "inlet":
            patch_entries += f"""
    {patch}
    {{
        type            variableHeightFlowRate;
        lowerBound      0.0;
        upperBound      1.0;
        value           uniform 0;
    }}
"""
        else:
            patch_entries += f"""
    {patch}
    {{
        type            zeroGradient;
    }}
"""

    content = f"""\
/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2406                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      alpha.water;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 0 0 0 0 0 0];

internalField   uniform 0;

boundaryField
{{{patch_entries}}}

// ************************************************************************* //
"""

    path = os.path.join(output_dir, "alpha.water")
    with open(path, "w") as f:
        f.write(content)
    print(f"Written: {path}")


def write_p_rgh(boundaries, output_dir="."):
    """Generate 0/p_rgh for interFoam."""

    os.makedirs(output_dir, exist_ok=True)

    patch_entries = ""
    for stl_name, bc_type, *_ in boundaries:
        patch = stl_name.replace(".stl", "").replace(".STL", "")

        if bc_type == "inlet":
            patch_entries += f"""
    {patch}
    {{
        type            fixedFluxPressure;
        value           uniform 0;
    }}
"""
        elif bc_type == "outlet":
            patch_entries += f"""
    {patch}
    {{
        type            totalPressure;
        p0              uniform 0;
        value           uniform 0;
    }}
"""
        else:
            patch_entries += f"""
    {patch}
    {{
        type            zeroGradient;
    }}
"""

    content = f"""\
/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2406                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      p_rgh;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [1 -1 -2 0 0 0 0];

internalField   uniform 0;

boundaryField
{{{patch_entries}}}

// ************************************************************************* //
"""

    path = os.path.join(output_dir, "p_rgh")
    with open(path, "w") as f:
        f.write(content)
    print(f"Written: {path}")


def write_g(vertical_axis, output_dir="."):
    """Generate constant/g."""

    os.makedirs(output_dir, exist_ok=True)

    axis = vertical_axis.lower().strip()
    if axis not in ("x", "y", "z"):
        raise ValueError(f"vertical_axis must be 'x', 'y', or 'z', got '{axis}'.")

    gx = -9.81 if axis == "x" else 0
    gy = -9.81 if axis == "y" else 0
    gz = -9.81 if axis == "z" else 0

    content = f"""\
/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2406                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       uniformDimensionedVectorField;
    object      g;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 1 -2 0 0 0 0];
value           ({gx} {gy} {gz});

// ************************************************************************* //
"""

    path = os.path.join(output_dir, "g")
    with open(path, "w") as f:
        f.write(content)
    print(f"Written: {path}")


def write_set_fields_dict(water_level, vertical_axis, output_dir="."):
    """Generate system/setFieldsDict."""

    os.makedirs(output_dir, exist_ok=True)

    axis = vertical_axis.lower().strip()
    if axis not in ("x", "y", "z"):
        raise ValueError(f"vertical_axis must be 'x', 'y', or 'z', got '{axis}'.")

    BIG  =  1e6
    SMAL = -1e6

    if axis == "x":
        box_min = f"({SMAL:.0f} {SMAL:.0f} {SMAL:.0f})"
        box_max = f"({water_level} {BIG:.0f} {BIG:.0f})"
    elif axis == "y":
        box_min = f"({SMAL:.0f} {SMAL:.0f} {SMAL:.0f})"
        box_max = f"({BIG:.0f} {water_level} {BIG:.0f})"
    else:
        box_min = f"({SMAL:.0f} {SMAL:.0f} {SMAL:.0f})"
        box_max = f"({BIG:.0f} {BIG:.0f} {water_level})"

    content = f"""\
/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2406                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      setFieldsDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

defaultFieldValues
(
    volScalarFieldValue alpha.water 0
);

regions
(
    boxToCell
    {{
        box {box_min} {box_max};
        fieldValues
        (
            volScalarFieldValue alpha.water 1
        );
    }}
);

// ************************************************************************* //
"""

    path = os.path.join(output_dir, "setFieldsDict")
    with open(path, "w") as f:
        f.write(content)
    print(f"Written: {path}")


def write_hRef(water_level, output_dir="."):
    """Generate constant/hRef."""

    os.makedirs(output_dir, exist_ok=True)

    content = f"""\
/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2406                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       uniformDimensionedScalarField;
    object      hRef;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 1 0 0 0 0 0];
value           {water_level};

// ************************************************************************* //
"""

    path = os.path.join(output_dir, "hRef")
    with open(path, "w") as f:
        f.write(content)
    print(f"Written: {path}")


def write_control_dict(writeInterval, purgeWrite, probes=None, write_iso_surface=True, output_dir="."):
    """Generate system/controlDict for interFoam."""

    os.makedirs(output_dir, exist_ok=True)

    if probes:
        probe_points = "\n".join(f"        ({x} {y} {z})" for x, y, z in probes)
        probes_block = f"""
    probes
    {{
        type            probes;
        libs            (sampling);
        writeControl    writeTime;
        fields          (U p_rgh);
        probeLocations
        (
{probe_points}
        );
    }}
"""
    else:
        probes_block = ""

    if write_iso_surface:
        iso_block = f"""
    freeSurface
    {{
        type            surfaces;
        libs            (sampling);
        writeControl    adjustableRunTime;
        writeInterval   {writeInterval};
        surfaceFormat   vtk;
        fields          (alpha.water);
        surfaces
        (
            isoSurface1
            {{
                type        isoSurface;
                isoField    alpha.water;
                isoValue    0.5;
                interpolate true;
            }}
        );
    }}
"""
    else:
        iso_block = ""

    functions_content = probes_block + iso_block

    content = f"""\
/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2406                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      controlDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

application     interFoam;

startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         200;

deltaT          0.001;

writeControl    adjustable;
writeInterval   {writeInterval};
purgeWrite      {purgeWrite};
writeFormat     ascii;
writePrecision  6;
writeCompression off;

timeFormat      general;
timePrecision   6;

runTimeModifiable yes;

adjustTimeStep  on;
maxCo           1;
maxAlphaCo      1;
maxDeltaT       1;

functions
{{{functions_content}}}

// ************************************************************************* //
"""

    path = os.path.join(output_dir, "controlDict")
    with open(path, "w") as f:
        f.write(content)
    print(f"Written: {path}")


def write_fv_solution(output_dir="."):
    """
    Generate system/fvSolution for interFoam with PIMPLE.
    Relaxation factors are set to 1 (no under-relaxation).
    """

    os.makedirs(output_dir, exist_ok=True)

    content = """\
/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2406                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSolution;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

solvers
{
    "alpha.water.*"
    {
        nAlphaCorr              1;
        nAlphaSubCycles         1;
        alphaOuterCorrectors    yes;
        cAlpha                  1;
        MULESCorr               yes;
        nLimiterIter            3;

        solver                  smoothSolver;
        smoother                symGaussSeidel;
        tolerance               1e-7;
        relTol                  0;
    }

    "pcorr.*"
    {
        solver                  GAMG;
        smoother                DIC;
        tolerance               1e-7;
        relTol                  0.0;
        nPreSweeps              0;
        nPostSweeps             2;
        nFinestSweeps           2;
        cacheAgglomeration      true;
        nCellsInCoarsestLevel   10;
        agglomerator            faceAreaPair;
        mergeLevels             1;
        maxIter                 5;
    }

    p_rgh
    {
        solver                  GAMG;
        smoother                DIC;
        tolerance               1e-7;
        relTol                  0.01;
        nPreSweeps              0;
        nPostSweeps             2;
        nFinestSweeps           2;
        cacheAgglomeration      true;
        nCellsInCoarsestLevel   10;
        agglomerator            faceAreaPair;
        mergeLevels             1;
    }

    p_rghFinal
    {
        $p_rgh;
        relTol                  0.0;
    }

    "(U|k|epsilon|omega).*"
    {
        solver                  smoothSolver;
        smoother                symGaussSeidel;
        nSweeps                 1;
        tolerance               1e-7;
        relTol                  0.1;
    }

    "(U|k|epsilon|omega)Final"
    {
        $U;
        relTol                  0;
    }
}

relaxationFactors
{
    fields
    {
        p_rgh               1;
    }
    equations
    {
        ".*"                1;
    }
}

PIMPLE
{
    momentumPredictor       no;
    nOuterCorrectors        1;
    nCorrectors             1;
    nNonOrthogonalCorrectors 1;
}

// ************************************************************************* //
"""

    path = os.path.join(output_dir, "fvSolution")
    with open(path, "w") as f:
        f.write(content)
    print(f"Written: {path}")


def write_fv_schemes(output_dir="."):
    """Generate system/fvSchemes for interFoam."""

    os.makedirs(output_dir, exist_ok=True)

    content = """\
/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2406                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      fvSchemes;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

ddtSchemes
{
    default         CrankNicolson 0.2;
}

gradSchemes
{
    default         cellLimited Gauss linear 1;
}

divSchemes
{
    div(rhoPhi,U)               Gauss linearUpwindV cellLimited Gauss linear 1;
    div(phi,alpha)              Gauss vanLeer;
    div(phirb,alpha)            Gauss interfaceCompression 1;
    div(phi,p_rgh)              Gauss linearUpwind default;
    div(phi,k)                  Gauss vanLeer;
    div(phi,epsilon)            Gauss vanLeer;
    div(phi,omega)              Gauss vanLeer;
    div(rhoPhi,k)               Gauss vanLeer;
    div(rhoPhi,omega)           Gauss vanLeer;
    div(rhoPhi,epsilon)         Gauss vanLeer;
    div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear;
}

laplacianSchemes
{
    default         Gauss linear limited 0.5;
}

interpolationSchemes
{
    default         linear;
}

snGradSchemes
{
    default         limited 0.5;
}

wallDist
{
    method          meshWave;
}

// ************************************************************************* //
"""

    path = os.path.join(output_dir, "fvSchemes")
    with open(path, "w") as f:
        f.write(content)
    print(f"Written: {path}")


def write_turbulence_properties(turbulence_model, output_dir="."):
    """
    Generate constant/turbulenceProperties.

    turbulence_model options:
      "kOmegaSST"    — RAS, k-omega SST
      "kEpsilon"     — RAS, standard k-epsilon
      "RNGkEpsilon"  — RAS, RNG k-epsilon
      "realizableKE" — RAS, Realizable k-epsilon
      "laminar"      — laminar (no turbulence model)
    """

    os.makedirs(output_dir, exist_ok=True)

    laminar_models = ("laminar",)

    if turbulence_model in laminar_models:
        sim_type = "laminar"
        ras_block = ""
    else:
        sim_type = "RAS"
        ras_block = f"""
RAS
{{
    RASModel        {turbulence_model};
    turbulence      on;
    printCoeffs     on;
}}
"""

    content = f"""\
/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2406                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      turbulenceProperties;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

simulationType  {sim_type};

densityVariable variable;
{ras_block}
// ************************************************************************* //
"""

    path = os.path.join(output_dir, "turbulenceProperties")
    with open(path, "w") as f:
        f.write(content)
    print(f"Written: {path}")


def write_k_kepsilon(boundaries, turbulence_U, turbulence_I, output_dir="."):
    """
    Generate 0/k for k-epsilon based models (kEpsilon, RNGkEpsilon, realizableKE).

    k = 1.5 * (U * I)^2

    Boundary conditions:
      inlet   -> fixedValue uniform <k>
      noSlip  -> kqRWallFunction
      slip    -> zeroGradient
      outlet  -> zeroGradient
    """

    os.makedirs(output_dir, exist_ok=True)

    k_value = 1.5 * (turbulence_U * turbulence_I) ** 2

    patch_entries = ""
    for stl_name, bc_type, *_ in boundaries:
        patch = stl_name.replace(".stl", "").replace(".STL", "")

        if bc_type == "inlet":
            patch_entries += f"""
    {patch}
    {{
        type            fixedValue;
        value           uniform {k_value:.6g};
    }}
"""
        elif bc_type == "noSlip":
            patch_entries += f"""
    {patch}
    {{
        type            kqRWallFunction;
        value           uniform {k_value:.6g};
    }}
"""
        else:
            # outlet, slip
            patch_entries += f"""
    {patch}
    {{
        type            zeroGradient;
    }}
"""

    content = f"""\
/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2406                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      k;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
// k = 1.5 * (U * I)^2 = 1.5 * ({turbulence_U} * {turbulence_I})^2 = {k_value:.6g} m2/s2

dimensions      [0 2 -2 0 0 0 0];

internalField   uniform {k_value:.6g};

boundaryField
{{{patch_entries}}}

// ************************************************************************* //
"""

    path = os.path.join(output_dir, "k")
    with open(path, "w") as f:
        f.write(content)
    print(f"Written: {path}  (k = {k_value:.6g} m²/s²)")


def write_k_komegasst(boundaries, turbulence_U, turbulence_I, output_dir="."):
    """
    Generate 0/k for kOmegaSST.

    k = 1.5 * (U * I)^2

    Boundary conditions:
      inlet   -> fixedValue uniform <k>
      noSlip  -> kLowReWallFunction (value is a placeholder, effectively zero at wall)
      slip    -> zeroGradient
      outlet  -> zeroGradient
    """

    os.makedirs(output_dir, exist_ok=True)

    k_value = 1.5 * (turbulence_U * turbulence_I) ** 2

    patch_entries = ""
    for stl_name, bc_type, *_ in boundaries:
        patch = stl_name.replace(".stl", "").replace(".STL", "")

        if bc_type == "inlet":
            patch_entries += f"""
    {patch}
    {{
        type            fixedValue;
        value           uniform {k_value:.6g};
    }}
"""
        elif bc_type == "noSlip":
            patch_entries += f"""
    {patch}
    {{
        type            kLowReWallFunction;
        value           uniform 1e-10;   // k at the wall is zero for low-Re
    }}
"""
        else:
            # outlet, slip
            patch_entries += f"""
    {patch}
    {{
        type            zeroGradient;
    }}
"""

    content = f"""\
/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2406                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      k;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
// k = 1.5 * (U * I)^2 = 1.5 * ({turbulence_U} * {turbulence_I})^2 = {k_value:.6g} m2/s2

dimensions      [0 2 -2 0 0 0 0];

internalField   uniform {k_value:.6g};

boundaryField
{{{patch_entries}}}

// ************************************************************************* //
"""

    path = os.path.join(output_dir, "k")
    with open(path, "w") as f:
        f.write(content)
    print(f"Written: {path}  (k = {k_value:.6g} m²/s²)")


def write_nut_kepsilon(boundaries, output_dir="."):
    """
    Generate 0/nut for k-epsilon based models (kEpsilon, RNGkEpsilon, realizableKE).

    noSlip + None  -> smooth wall functions (nutkWallFunction) — resolved walls
                      not typical for k-epsilon, defaults to smooth
    noSlip + 0     -> nutkWallFunction (smooth)
    noSlip + Ks>0  -> nutkRoughWallFunction, Ks=<value>, Cs=0.5 (fixed default)
    all others     -> calculated
    """

    os.makedirs(output_dir, exist_ok=True)

    patch_entries = ""
    for entry in boundaries:
        stl_name = entry[0]
        bc_type  = entry[1]
        ks       = entry[4] if len(entry) > 4 else None
        patch    = stl_name.replace(".stl", "").replace(".STL", "")

        if bc_type == "noSlip":
            if ks is not None and ks > 0:
                patch_entries += f"""
    {patch}
    {{
        type            nutkRoughWallFunction;
        Ks              uniform {ks};
        Cs              uniform 0.5;
        value           uniform 0;
    }}
"""
            else:
                # ks=None or ks=0 -> smooth wall functions
                patch_entries += f"""
    {patch}
    {{
        type            nutkWallFunction;
        value           uniform 0;
    }}
"""
        else:
            patch_entries += f"""
    {patch}
    {{
        type            calculated;
        value           uniform 0;
    }}
"""

    content = f"""\
/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2406                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      nut;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 2 -1 0 0 0 0];

internalField   uniform 0;

boundaryField
{{{patch_entries}}}

// ************************************************************************* //
"""

    path = os.path.join(output_dir, "nut")
    with open(path, "w") as f:
        f.write(content)
    print(f"Written: {path}")


def write_nut_komegasst(boundaries, output_dir="."):
    """
    Generate 0/nut for kOmegaSST.

    noSlip + None  -> resolve wall (low-Re) -> nutLowReWallFunction
    noSlip + 0     -> smooth wall functions -> nutkWallFunction
    noSlip + Ks>0  -> rough wall functions  -> nutkRoughWallFunction, Ks=<value>, Cs=0.5
    all others     -> calculated
    """

    os.makedirs(output_dir, exist_ok=True)

    patch_entries = ""
    for entry in boundaries:
        stl_name = entry[0]
        bc_type  = entry[1]
        ks       = entry[4] if len(entry) > 4 else None
        patch    = stl_name.replace(".stl", "").replace(".STL", "")

        if bc_type == "noSlip":
            if ks is None:
                # noSlip + None = resolve the wall (low-Re, no wall functions)
                patch_entries += f"""
    {patch}
    {{
        type            nutLowReWallFunction;
        value           uniform 0;
    }}
"""
            elif ks == 0 or ks == 0.0:
                # smooth wall functions
                patch_entries += f"""
    {patch}
    {{
        type            nutkWallFunction;
        value           uniform 0;
    }}
"""
            else:
                # rough wall functions
                patch_entries += f"""
    {patch}
    {{
        type            nutkRoughWallFunction;
        Ks              uniform {ks};
        Cs              uniform 0.5;
        value           uniform 0;
    }}
"""
        else:
            patch_entries += f"""
    {patch}
    {{
        type            calculated;
        value           uniform 0;
    }}
"""

    content = f"""\
/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2406                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      nut;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

dimensions      [0 2 -1 0 0 0 0];

internalField   uniform 0;

boundaryField
{{{patch_entries}}}

// ************************************************************************* //
"""

    path = os.path.join(output_dir, "nut")
    with open(path, "w") as f:
        f.write(content)
    print(f"Written: {path}")


def write_epsilon(boundaries, turbulence_U, turbulence_I, turbulence_L, output_dir="."):
    """
    Generate 0/epsilon for k-epsilon based models (kEpsilon, RNGkEpsilon, realizableKE).

    k   = 1.5 * (U * I)^2
    eps = Cmu^(3/4) * k^(3/2) / L    (Cmu = 0.09)

    Boundary conditions:
      inlet   -> fixedValue uniform <epsilon>
      noSlip  -> epsilonWallFunction
      all others -> zeroGradient
    """

    os.makedirs(output_dir, exist_ok=True)

    Cmu     = 0.09
    k_value = 1.5 * (turbulence_U * turbulence_I) ** 2
    eps     = (Cmu ** 0.75) * (k_value ** 1.5) / turbulence_L

    patch_entries = ""
    for stl_name, bc_type, *_ in boundaries:
        patch = stl_name.replace(".stl", "").replace(".STL", "")

        if bc_type == "inlet":
            patch_entries += f"""
    {patch}
    {{
        type            fixedValue;
        value           uniform {eps:.6g};
    }}
"""
        elif bc_type == "noSlip":
            patch_entries += f"""
    {patch}
    {{
        type            epsilonWallFunction;
        value           uniform {eps:.6g};
    }}
"""
        else:
            patch_entries += f"""
    {patch}
    {{
        type            zeroGradient;
    }}
"""

    content = f"""\
/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2406                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      epsilon;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
// k   = 1.5 * (U * I)^2 = {k_value:.6g} m2/s2
// eps = Cmu^(3/4) * k^(3/2) / L = 0.09^0.75 * {k_value:.6g}^1.5 / {turbulence_L} = {eps:.6g} m2/s3

dimensions      [0 2 -3 0 0 0 0];

internalField   uniform {eps:.6g};

boundaryField
{{{patch_entries}}}

// ************************************************************************* //
"""

    path = os.path.join(output_dir, "epsilon")
    with open(path, "w") as f:
        f.write(content)
    print(f"Written: {path}  (epsilon = {eps:.6g} m²/s³)")


def write_omega(boundaries, turbulence_U, turbulence_I, turbulence_L, output_dir="."):
    """
    Generate 0/omega for kOmegaSST.

    k     = 1.5 * (U * I)^2
    omega = k^(1/2) / (Cmu^(1/4) * L)    (Cmu = 0.09)

    Boundary conditions:
      inlet   -> fixedValue uniform <omega>
      noSlip  -> omegaWallFunction
      all others -> zeroGradient
    """

    os.makedirs(output_dir, exist_ok=True)

    Cmu     = 0.09
    k_value = 1.5 * (turbulence_U * turbulence_I) ** 2
    omega   = (k_value ** 0.5) / ((Cmu ** 0.25) * turbulence_L)

    patch_entries = ""
    for stl_name, bc_type, *_ in boundaries:
        patch = stl_name.replace(".stl", "").replace(".STL", "")

        if bc_type == "inlet":
            patch_entries += f"""
    {patch}
    {{
        type            fixedValue;
        value           uniform {omega:.6g};
    }}
"""
        elif bc_type == "noSlip":
            patch_entries += f"""
    {patch}
    {{
        type            omegaWallFunction;
        value           uniform {omega:.6g};
    }}
"""
        else:
            patch_entries += f"""
    {patch}
    {{
        type            zeroGradient;
    }}
"""

    content = f"""\
/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2406                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       volScalarField;
    object      omega;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
// k     = 1.5 * (U * I)^2 = {k_value:.6g} m2/s2
// omega = k^(1/2) / (Cmu^(1/4) * L) = {k_value:.6g}^0.5 / (0.09^0.25 * {turbulence_L}) = {omega:.6g} 1/s

dimensions      [0 0 -1 0 0 0 0];

internalField   uniform {omega:.6g};

boundaryField
{{{patch_entries}}}

// ************************************************************************* //
"""

    path = os.path.join(output_dir, "omega")
    with open(path, "w") as f:
        f.write(content)
    print(f"Written: {path}  (omega = {omega:.6g} s⁻¹)")


def write_transport_properties(output_dir="."):
    """Generate constant/transportProperties for interFoam (water/air)."""

    os.makedirs(output_dir, exist_ok=True)

    content = """\
/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2406                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      transportProperties;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

phases          (water air);

water
{
    transportModel  Newtonian;
    nu              1e-06;
    rho             1000;
}

air
{
    transportModel  Newtonian;
    nu              1.48e-05;
    rho             1;
}

sigma           0.07;

// ************************************************************************* //
"""

    path = os.path.join(output_dir, "transportProperties")
    with open(path, "w") as f:
        f.write(content)
    print(f"Written: {path}")


def write_decompose_par_dict(n_subdomains, output_dir="."):
    """Generate system/decomposeParDict. Method hardcoded to scotch."""

    os.makedirs(output_dir, exist_ok=True)

    content = f"""\
/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v2406                                 |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      decomposeParDict;
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

numberOfSubdomains {n_subdomains};

method          scotch;

// ************************************************************************* //
"""

    path = os.path.join(output_dir, "decomposeParDict")
    with open(path, "w") as f:
        f.write(content)
    print(f"Written: {path}  ({n_subdomains} subdomains)")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    write_block_mesh_dict(output_dir_system)
    write_snappy_hex_mesh_dict(stl_files, locationInMesh, boundaries, refinement_regions, vertical_axis, output_dir_system)
    write_surface_feature_extract_dict(stl_files, output_dir_system)
    write_U(boundaries, output_dir_zero)
    write_alpha_water(boundaries, output_dir_zero)
    write_p_rgh(boundaries, output_dir_zero)
    write_g(vertical_axis, output_dir_constant)
    write_hRef(water_level, output_dir_constant)
    write_control_dict(writeInterval, purgeWrite, probes, write_iso_surface, output_dir_system)
    write_set_fields_dict(water_level, vertical_axis, output_dir_system)
    write_fv_solution(output_dir_system)
    write_fv_schemes(output_dir_system)
    write_turbulence_properties(turbulence_model, output_dir_constant)
    write_transport_properties(output_dir_constant)
    write_decompose_par_dict(n_subdomains, output_dir_system)

    # --- turbulence field files ---
    if turbulence_model in ("kEpsilon", "RNGkEpsilon", "realizableKE"):
        write_k_kepsilon(boundaries, turbulence_U, turbulence_I, output_dir_zero)
        write_nut_kepsilon(boundaries, output_dir_zero)
        write_epsilon(boundaries, turbulence_U, turbulence_I, turbulence_L, output_dir_zero)
    elif turbulence_model == "kOmegaSST":
        write_k_komegasst(boundaries, turbulence_U, turbulence_I, output_dir_zero)
        write_nut_komegasst(boundaries, output_dir_zero)
        write_omega(boundaries, turbulence_U, turbulence_I, turbulence_L, output_dir_zero)
