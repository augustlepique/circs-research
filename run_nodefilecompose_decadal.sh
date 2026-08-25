#!/usr/bin/env bash
#
# run_nodefilecompose_decadal.sh
#
# Decadal cyclone-relative composites for the OC (outbreak) and NOC 
# (nonoutbreak) ETC populations, DJFM 1996-2025. 
#
# This is similar to run_nodefilecompose.sh with one extra axis. Instead of one composite
# per (population, variable) we produce one per (population, DECADE, variable)
# 
# Every composite-grid parameter below is identical to run_nodefilecompose.sh
# 
# pressure level index map:
#   idx  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20
#   hPa 1000 975 950 925 900 850 800 750 700 650 600 550 500 450 400 350 300 250 200 150 100
# 
# Usage:  nohup ./run_nodefilecompose_decadal.sh > {something}.log 2>&1 & (for my own reference)
####################################################################################################

set -euo pipefail

# Paths
NODEDIR="/home1/lepique/circs_research/data" # where the nodefiles live
DATADIR="/data1/lepique/era5_TE_composite"   # where the in_data_list files live
OUTDIR="/data1/lepique/composites_decadal"   # directory for the decadal composites

WORKDIR="${OUTDIR}/decadal_inputs"           # generated sub-nodefiles + sub-lists

mkdir -p "${OUTDIR}" "${WORKDIR}"

#OC_NODEFILE="${NODEDIR}/outbreak_cyclones_min_msl_cores_filtered.txt"
OC_NODEFILE="${NODEDIR}/outbreak_cyclones_time_maxreports_cores.txt"
NOC_NODEFILE="${NODEDIR}/nonoutbreak_cyclones_min_msl_cores_filtered.txt"

SFC_LIST="${DATADIR}/in_data_list_sfc.txt"       # 3294 files, 1995-12-01 .. 2025-04-01
PLEV_LIST="${DATADIR}/in_data_list_plev.txt"     # 3294 files, same span
DERIV_LIST="${DATADIR}/in_data_list_derived.txt" # does not exist yet; deriv vars
                                                 # stay commented out below

# Composite grid parameters
DX=1.0
RESX=80
TIMEDELTA="1h"
INFMT="lon,lat,msl,phis"
REF="time_maxreports"

# Decade definition 
# tag is season range: "LO-HI" inclusive, in DJFM season-years (Dec belongs to following year's JFM)
# To change from 3 to 2 samples, replace the following with :
#     DECADES=( "1996-2010" "2011-2025" )
# ---------------------------------------------------------------------------
DECADES=(
    "1996-2005"
    "2006-2015"
    "2016-2025"
)

# ---------------------------------------------------------------------------
# Variable table.  Format per entry:  "VAROUT:VARSPEC:LIST"
#   VAROUT  = output filename stem (e.g. z500)
#   VARSPEC = the --var argument, WITH level index for plev vars (e.g. z(12))
#   LIST    = which data list to use:  sfc | plev | deriv
# ---------------------------------------------------------------------------
VARIABLES=(
    # --- 500 hPa: geopotential + relative vorticity ---
    "z500:z(12):plev"
    "vo500:vo(12):plev"

    # --- 250 hPa: geopotential + winds ---
    "z250:z(17):plev"
    "u250:u(17):plev"
    "v250:v(17):plev"

    # --- 850 hPa winds, temperature, and moisture ---
    "u850:u(5):plev"
    "v850:v(5):plev"
    "t850:t(5):plev"
    "q850:q(5):plev"

    # --- surface / single-level ---
    "msl:msl:sfc"
    "t2m:t2m:sfc"
    "d2m:d2m:sfc"
    "tcwv:tcwv:sfc"
    "cape:cape:sfc"

    # --- derived (compute_derived_composite.py) -- needs in_data_list_derived.txt
    # "thetae850:thetae(5):deriv"
    # "shear06:shear06:deriv"
    # "shear03:shear03:deriv"
)

# ---------------------------------------------------------------------------
# Populations:  "TAG:NODEFILE"
# ---------------------------------------------------------------------------
POPULATIONS=(
    "oc:${OC_NODEFILE}"
    #"noc:${NOC_NODEFILE}"  # commented out for using REF="time_maxreports"
)

# ---------------------------------------------------------------------------
# make_decade_nodefile: subset a StitchNodes nodefile to one season range.
#
# Format written by write_snapshot_nodefile() in TE_compositing.ipynb:
#     start <n> <YYYY> <MM> <DD> <HH>
#     \t<i> <j> <lon> <lat> <msl> <phis> <YYYY> <MM> <DD> <HH>
# i.e. a header line plus n node lines (n == 1 here, one snapshot per cyclone,
# at the min-MSL (or other) reference time). The loop over n keeps this correct if you ever
# feed it a multi-node nodefile.
#
# The season decision is made once on the header line and then applied to the
# whole block, so a block is never half-copied.
# ---------------------------------------------------------------------------
make_decade_nodefile () {
    local src="$1" lo="$2" hi="$3" dst="$4"

    awk -v lo="${lo}" -v hi="${hi}" '
        $1 == "start" {
            n = $2 + 0                          # node count for this block
            y = $3 + 0; m = $4 + 0              # header date: YYYY MM DD HH
            season = (m == 12) ? y + 1 : y      # Dec -> following JFM season
            keep = (season >= lo && season <= hi)
            if (keep) print
            for (k = 0; k < n; k++)
                if ((getline nodeline) > 0 && keep) print nodeline
            next
        }
    ' "${src}" > "${dst}"
}

# ---------------------------------------------------------------------------
# make_decade_datalist: subset an in_data_list to the same season range.
# Date is taken from the filename (era5_plev_YYYY-MM-DD.nc), with the same
# Dec -> next-season rule so list and nodefile always agree.

make_decade_datalist () {
    local src="$1" lo="$2" hi="$3" dst="$4"

    awk -v lo="${lo}" -v hi="${hi}" '
        match($0, /[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]/) {
            d = substr($0, RSTART, RLENGTH)
            y = substr(d, 1, 4) + 0; m = substr(d, 6, 2) + 0
            season = (m == 12) ? y + 1 : y
            if (season >= lo && season <= hi) print
        }
    ' "${src}" > "${dst}"
}

# ---------------------------------------------------------------------------
# Phase 1: build all decade inputs and print the sample sizes
#
# Done up front, before any compositing, to see n per (pop, decade) and abort if a bin
# is too thin.
echo "====================== BUILDING DECADE INPUTS ========================"
printf "%-6s %-12s %8s\n" "POP" "DECADE" "N_CYC"

for pop_entry in "${POPULATIONS[@]}"; do
    pop_tag="${pop_entry%%:*}"
    nodefile="${pop_entry#*:}"

    for dec_tag in "${DECADES[@]}"; do
        lo="${dec_tag%%-*}"
        hi="${dec_tag##*-}"

        sub_nodefile="${WORKDIR}/${pop_tag}_${dec_tag}_${REF}_nodes.txt"
        make_decade_nodefile "${nodefile}" "${lo}" "${hi}" "${sub_nodefile}"

        # '|| true' because grep -c exits 1 on zero matches, which set -e would
        # otherwise treat as a fatal error on an empty decade.
        n_cyc=$(grep -c '^start' "${sub_nodefile}" || true)
        printf "%-6s %-12s %8s\n" "${pop_tag}" "${dec_tag}" "${n_cyc}"
    done 
done 

# Data lists are population-independent, so build them once per decade.
for dec_tag in "${DECADES[@]}"; do 
    lo="${dec_tag%%-*}"
    hi="${dec_tag##*-}"
    make_decade_datalist "${SFC_LIST}" "${lo}" "${hi}" "${WORKDIR}/list_sfc_${dec_tag}.txt"
    make_decade_datalist "${PLEV_LIST}" "${lo}" "${hi}" "${WORKDIR}/list_plev_${dec_tag}.txt"
    [[ -f "${DERIV_LIST}" ]] && \
        make_decade_datalist "${DERIV_LIST}" "${lo}" "${hi}" "${WORKDIR}/list_deriv_${dec_tag}.txt"
    
    echo "  ${dec_tag}: $(wc -l < "${WORKDIR}/list_sfc_${dec_tag}.txt") sfc / "\
"$(wc -l < "${WORKDIR}/list_plev_${dec_tag}.txt") plev files"
done 

# ------------------------------------------------------------------------
# Compose
#
#
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

# ------------------------------------------------------------------------
# Phase 2 --driver: populations x decades x variables
#
# 2 populations x 3 decades x 14 variables = 84 output files. [skip] guard in compose()
# makes this restartable
# ------------------------------------------------------------------------
for pop_entry in "${POPULATIONS[@]}"; do
    pop_tag="${pop_entry%%:*}"

    for dec_tag in "${DECADES[@]}"; do
        sub_nodefile="${WORKDIR}/${pop_tag}_${dec_tag}_${REF}_nodes.txt"

        # An empty decade would make NodeFileCompose fail or emit an all-NaN
        # field that silently poisons a difference map. Skip it loudly instead.
        n_cyc=$(grep -c '^start' "${sub_nodefile}" || true)
        if [[ "${n_cyc}" -eq 0 ]]; then
            echo "!!! ${pop_tag} ${dec_tag}: no cyclones, skipping"
            continue
        fi

        echo "==================== ${pop_tag^^}  ${dec_tag}  (n=${n_cyc}) ===================="

        for var_entry in "${VARIABLES[@]}"; do
            IFS=":" read -r varout varspec listtag <<< "${var_entry}"

            if [[ "${listtag}" == "plev" ]]; then
                datalist="${WORKDIR}/list_plev_${dec_tag}.txt"
            elif [[ "${listtag}" == "deriv" ]]; then
                datalist="${WORKDIR}/list_deriv_${dec_tag}.txt"
            else
                datalist="${WORKDIR}/list_sfc_${dec_tag}.txt"
            fi

            out="${OUTDIR}/${pop_tag}_${dec_tag}_${REF}_${varout}.nc"
            compose "${sub_nodefile}" "${datalist}" "${varspec}" "${out}"
        done
    done
done

echo "==================== DONE ===================="
echo "Decadal composites (mean + per-cyclone snapshots) written to ${OUTDIR}"
echo "Naming: <pop>_<season_range>_${REF}_<var>.nc"







