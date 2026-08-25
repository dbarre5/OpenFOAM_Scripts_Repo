#!/bin/bash
set -e

DICT="system/snappyHexMeshDict"
LOGDIR="results/logs"
CSV="results/runs.csv"

mkdir -p "$LOGDIR"

# Parameters passed as env vars
: "${featureAngle:?}"
: "${nGrow:?}"
: "${nRelaxIter:?}"
: "${nSmoothSurfaceNormals:?}"
: "${nSmoothNormals:?}"
: "${nSmoothThickness:?}"

RUN_ID=$(date +%s%N)
OUT="$LOGDIR/snappy_$RUN_ID.out"

# Update dict
set_param () {
    sed -i -E "s|($1[[:space:]]+)[^;]+;|\1$2;|g" "$DICT"
}

set_param featureAngle "$featureAngle"
set_param nGrow "$nGrow"
set_param nRelaxIter "$nRelaxIter"
set_param nSmoothSurfaceNormals "$nSmoothSurfaceNormals"
set_param nSmoothNormals "$nSmoothNormals"
set_param nSmoothThickness "$nSmoothThickness"

# Run snappy
# Run snappy
# Run snappy
rm -rf 0.0001
if ! snappyHexMesh > "$OUT" 2>&1; then
    echo "snappyHexMesh failed" >&2
    exit 1
fi

coverage=$(awk '
# Detect table header
/^patch[[:space:]]+faces[[:space:]]+layers[[:space:]]+overall[[:space:]]+thickness/ {header1=1; next}
header1 && /^[[:space:]]*target[[:space:]]+mesh/ {header2=1; next}
header2 && /^[-]+[[:space:]]+[-]+/ {inblock=1; next}

# Stop if we reach a line that is not a patch data line
inblock && (NF < 2 || $1 !~ /^[[:alnum:]_]+$/) {exit}

# Process table data lines
inblock {
    gsub(/^[ \t]+|[ \t]+$/, "", $0)
    n = split($0, a)
    patch = a[1]
    faces = a[2]
    cov = a[n]/100        # last column [%] → fraction
    sum += faces*cov
    tot += faces
}

END {
    if(tot>0) printf "%.3f", sum/tot
    else print "NA"
}
' "$OUT")

if [[ -z "$coverage" || "$coverage" == "NA" ]]; then
    echo "coverage extraction failed" >&2
    exit 1
fi

echo "$featureAngle,$nGrow,$nRelaxIter,$nSmoothSurfaceNormals,$nSmoothNormals,$nSmoothThickness,$coverage" >> "$CSV"

echo "$coverage"

