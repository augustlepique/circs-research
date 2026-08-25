#!/usr/bin/env bash

# run_nodefilecompose.sh

# Cyclone relative composites for OC (outbreak) and
# NOC (nonoutbreak) ETC populations, full 30-year dataset. 

# runs NodeFileCompose on raw ERA5 files (no NodeFileFilter step)

# Presure level index map for reference:
#   idx  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20
#   hPa 1000 975 950 925 900 850 800 750 700 650 600 550 500 450 400 350 300 250 200 150 100
#
# So:  850 -> (5)   500 -> (12)   300 -> (16)   200 -> (18)
#
# variables are composited separately
##########################################################################3

set -euo pipefail

# Paths:
NODEDIR="/home1/lepique/circs_research/data"     # where the nodefiles live
DATADIR="/data1/lepique/era5_TE_composite"       # where in_data_list files live
OUTDIR="/data1/lepique/composites_full_filtered"
mkdir -p "${OUTDIR}"

OC_NODEFILE="${NODEDIR}/outbreak_cyclones_min_msl_cores_filtered.txt"
NOC_NODEFILE="${NODEDIR}/nonoutbreak_cyclones_min_msl_cores_filtered.txt"

SFC_LIST="${DATADIR}/in_data_list_sfc.txt"
PLEV_LIST="${DATADIR}/in_data_list_plev.txt"
DERIV_LIST="${DATADIR}/in_data_list_derived.txt"   # derived vars (theta-e, shear)

#
# Composite grid parameters
#
DX=1.0
RESX=80
TIMEDELTA="1h"
INFMT="lon,lat,msl,phis"

# ---------------------------------------------------------------------------
# Variable table.  Format per entry:  "VAROUT:VARSPEC:LIST"
#   VAROUT  = output filename stem (e.g. z500)
#   VARSPEC = the --var argument, WITH level index for plev vars (e.g. z(12))
#   LIST    = which data list to use:  sfc | plev | deriv
#
# Add or remove a line to change the variable set — nothing else to edit.
# ---------------------------------------------------------------------------

VARIABLES=(
    # --- 500 hPa: geopotential + relative vorticity ---
    "z500:z(12):plev"
    "vo500:vo(12):plev"

    # --- 250 hPa: geopotential + winds ---
    "z250:z(17):plev"
    "u250:u(17):plev"
    "v250:v(17):plev"

    # --- 850 hPa winds, temperature, moisture, and geopotential height ---
    "u850:u(5):plev"
    "v850:v(5):plev"
    "t850:t(5):plev"
    "q850:q(5):plev"
    "z850:z(5):plev"

    # --- surface / single-level ---
    "msl:msl:sfc"
    "t2m:t2m:sfc"
    "d2m:d2m:sfc"
    "tcwv:tcwv:sfc"
    "cape:cape:sfc"

    # --- derived (compute_derived_composite.py) ---
    # theta-e is 3-D: add more thetae(idx) lines for other levels as needed.
    "thetae850:thetae(5):deriv"
    "shear06:shear06:deriv"
    "shear06_u:shear06_u:deriv"
    "shear06_v:shear06_v:deriv"
    "shear03:shear03:deriv"
    "shear03_u:shear03_u:deriv"
    "shear03_v:shear03_v:deriv"
    "shear01:shear01:deriv"
    "shear01_u:shear01_u:deriv"
    "shear01_v:shear01_v:deriv"
    
)

# ---------------------------------------------------------------------------
# Populations:  "TAG:NODEFILE"
# ---------------------------------------------------------------------------
POPULATIONS=(
    "oc:${OC_NODEFILE}"
    "noc:${NOC_NODEFILE}"
)

# (aided by AI for the following code)
#
# --op mean and --snapshots coexist in one file: the output holds both the
# composite mean <var>(y, x) AND the per-cyclone stack snap_<var>(snapshot, y, x)
# (verified: nanmean(snap_<var>, axis=0) reproduces <var>). The stack is what the
# permutation significance test reshuffles; identify each snapshot by snap_pathid
# (0-indexed, nodefile order) since snap_time is a fill value in this build.
# --snapshots is a bare presence flag 
compose () {
    local nodefile="$1" datalist="$2" varspec="$3" out="$4"
    if [[ -f "${out}" ]]; then
        echo "  [skip] $(basename "${out}") exists"
        return
    fi
    echo "  >>> ${varspec}  ->  $(basename "${out}")"
    NodeFileCompose \
        --in_nodefile      "${nodefile}" \
        --in_nodefile_type "SN" \
        --in_fmt           "${INFMT}" \
        --in_data_list     "${datalist}" \
        --out_data         "${out}" \
        --var              "${varspec}" \
        --op               "mean" \
        --snapshots \
        --max_time_delta   "${TIMEDELTA}" \
        --dx               "${DX}" \
        --resx             "${RESX}" \
        --lonname          "longitude" \
        --latname          "latitude" \
        --regional
}


# ---------------------------------------------------------------------------
# Driver: loop populations x variables
# ---------------------------------------------------------------------------
for pop_entry in "${POPULATIONS[@]}"; do
    pop_tag="${pop_entry%%:*}"
    nodefile="${pop_entry#*:}"
    echo "==================== ${pop_tag^^} ===================="

    for var_entry in "${VARIABLES[@]}"; do
        IFS=":" read -r varout varspec listtag <<< "${var_entry}"

        if [[ "${listtag}" == "plev" ]]; then
            datalist="${PLEV_LIST}"
        elif [[ "${listtag}" == "deriv" ]]; then
            datalist="${DERIV_LIST}"
        else
            datalist="${SFC_LIST}"
        fi

        out="${OUTDIR}/${pop_tag}_min_msl_${varout}.nc"
        compose "${nodefile}" "${datalist}" "${varspec}" "${out}"
    done
done

echo "==================== DONE ===================="
echo "Composites (mean + per-cyclone snapshots) written to ${OUTDIR}"
