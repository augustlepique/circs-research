#!/usr/bin/env bash

# run_nodefilecompose_snow_vsneither.sh

# Cyclone relative composites for the four way snow x outbreak ETC populations BUT controls 
# for intensity and seasonality by having matched each cyclone from the 3 groups to a cyclone
# from the NEITHER group

#   SNOW_ONLY
#   OC_SNOW
#   OC_ONLY
#   NEITHER
#
# Same reference-node selection as run_nodefilecompose.sh (DJFM, min MSL
# reference time, CONUS box filter). 

# runs NodeFileCompose on the raw ERA5 files

# pressure level index map for reference:
#   idx  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20
#   hPa 1000 975 950 925 900 850 800 750 700 650 600 550 500 450 400 350 300 250 200 150 100
#
# variables are composited separately


set -euo pipefail

# Paths:
NODEDIR="/home1/lepique/circs_research/data"     # where the nodefiles live
DATADIR="/data1/lepique/era5_TE_composite"       # where in_data_list files live
OUTDIR="/data1/lepique/composites_snow_groups_vsneither"
mkdir -p "${OUTDIR}"

SFC_LIST="${DATADIR}/in_data_list_sfc.txt"
PLEV_LIST="${DATADIR}/in_data_list_plev.txt"
DERIV_LIST="${DATADIR}/in_data_list_derived.txt"   # derived vars (theta-e, shear)
IRROT_LIST="${DATADIR}/in_data_list_irrot.txt"     # irrotational wind + PV advection (compute_irrotational.py)
PV_LIST="${DATADIR}/in_data_list_pv.txt"           # PV, all 21 levels (download_era5_composite.py --only pv)
W_LIST="${DATADIR}/in_data_list_w.txt"             # omega, all 21 levels (--only w)


# Composite grid parameters
DX=1.0
RESX=80
TIMEDELTA="1h"
INFMT="lon,lat,msl,phis"

# ---------------------------------------------------------------------------
# Variable table.  Format per entry:  "VAROUT:VARSPEC:LIST"
#   VAROUT  = output filename stem (e.g. z500)
#   VARSPEC = the --var argument, WITH level index for plev vars (e.g. z(12))
#   LIST    = which data list to use:  sfc | plev | deriv | irrot | pv | w
#

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
    
    # In era5_TE_composite, pv and w are in their own era5_pv_ / era5_w_ files
    # (not the plev files); both hold all 21 levels, so the plev index map applies.
    "pv850:pv(5):pv"           # low-level (diabatically generated) PV
    "pv500:pv(12):pv"          # mid-level PV
    "pv300:pv(16):pv"          # upper-level PV (tropopause / stratospheric intrusions)
    "w700:w(8):w"              # omega, Pa s-1 (negative = ascent)
    "w500:w(12):w"
    # cp is not downloaded for era5_TE_composite (only the onset sfc files have it)
    #"cp:cp:sfc"               # convective precip, m, 1-h accumulation ending at valid time

    # --- irrotational wind + PV advection by it (compute_irrotational.py) ---
    # 3-D irrot fields hold only 300/250/200 hPa: idx 0 = 300, 1 = 250, 2 = 200
    "uchi_l:uchi_l:irrot"               # 300-200 hPa layer means
    "vchi_l:vchi_l:irrot"
    "pv_l:pv_l:irrot"
    "pvadv_chi_l:pvadv_chi_l:irrot"     # K m2 kg-1 s-2; x 1e6 x 86400 = PVU/day
    "chi250:chi(1):irrot"               # 250 hPa velocity potential
    "uchi250:uchi(1):irrot"
    "vchi250:vchi(1):irrot"
    "pvadv_chi250:pvadv_chi(1):irrot"

)

# Populations:  
# one population per side of each "<group> vs NEITHER" match, written by
# TE_compositing.ipynb as snowgrp_vsneither_<group>_vs_neither__<side>.txt
POPULATIONS=()
shopt -s nullglob
for nf in "${NODEDIR}"/snowgrp_vsneither_*.txt; do
    tag="$(basename "${nf}" .txt)"
    POPULATIONS+=("${tag#snowgrp_vsneither_}:${nf}")
done
shopt -u nullglob
if (( ${#POPULATIONS[@]} != 6 )); then
    echo "ERROR: expected 6 matched nodefiles in ${NODEDIR}, found ${#POPULATIONS[@]}" >&2
    exit 1
fi

# fail early if a nodefile hasn't been written yet (TE_compositing.ipynb)
for pop_entry in "${POPULATIONS[@]}"; do
    nodefile="${pop_entry#*:}"
    if [[ ! -s "${nodefile}" ]]; then
        echo "ERROR: missing or empty nodefile ${nodefile}" >&2
        exit 1
    fi
done

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

# Driver: loop populations x variables
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
        elif [[ "${listtag}" == "irrot" ]]; then
            datalist="${IRROT_LIST}"
        elif [[ "${listtag}" == "pv" ]]; then
            datalist="${PV_LIST}"
        elif [[ "${listtag}" == "w" ]]; then
            datalist="${W_LIST}"
        else
            datalist="${SFC_LIST}"
        fi

        out="${OUTDIR}/${pop_tag}_min_msl_${varout}.nc"
        compose "${nodefile}" "${datalist}" "${varspec}" "${out}"
    done
done

echo "==================== DONE ===================="
echo "Composites (mean + per-cyclone snapshots) written to ${OUTDIR}"



