#!/usr/bin/env bash

# run_nodefilecompose_matched.sh
#
# Regenerates composites for the intensity-matched NOC population (1:2
# nearest neighbor match on min_msl against OCs). OC's population is unchanged
# by matching, so its existing composites in composites_full_filtered are
# copied over rather than recomputed -- only NOCs need a fresh NodeFileCompose
# pass. Otherwise identical to run_nodefilecompose.sh

set -euo pipefail

# Paths:
NODEDIR="/home1/lepique/circs_research/data"
DATADIR="/data1/lepique/era5_TE_composite"
SRCDIR="/data1/lepique/composites_full_filtered"     # existing OC composites (unchanged population)
OUTDIR="/data1/lepique/composites_matched1to2"
mkdir -p "${OUTDIR}"

NOC_NODEFILE="${NODEDIR}/nonoutbreak_cyclones_min_msl_cores_matched1to2.txt"

SFC_LIST="${DATADIR}/in_data_list_sfc.txt"
PLEV_LIST="${DATADIR}/in_data_list_plev.txt"
DERIV_LIST="${DATADIR}/in_data_list_derived.txt"

DX=1.0
RESX=80
TIMEDELTA="1h"
INFMT="lon,lat,msl,phis"

# Same variable table as run_nodefilecompose.sh
VARIABLES=(
    "z500:z(12):plev"
    "vo500:vo(12):plev"
    "z250:z(17):plev"
    "u250:u(17):plev"
    "v250:v(17):plev"
    "u850:u(5):plev"
    "v850:v(5):plev"
    "t850:t(5):plev"
    "q850:q(5):plev"
    "z850:z(5):plev"
    "msl:msl:sfc"
    "t2m:t2m:sfc"
    "d2m:d2m:sfc"
    "tcwv:tcwv:sfc"
    "cape:cape:sfc"
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


echo "==================== copying unchanged OC composites ===================="
for var_entry in "${VARIABLES[@]}"; do
    IFS=":" read -r varout _ _ <<< "${var_entry}"
    src="${SRCDIR}/oc_min_msl_${varout}.nc"
    dst="${OUTDIR}/oc_min_msl_${varout}.nc"
    if [[ -f "${dst}" ]]; then
        echo "  [skip] $(basename "${dst}") exists"
    elif [[ -f "${src}" ]]; then
        cp "${src}" "${dst}"
        echo "  [copy] $(basename "${src}")"
    else
        echo "  !! MISSING source: ${src}"
    fi
done

echo "==================== NOC (matched 1:2) ===================="
for var_entry in "${VARIABLES[@]}"; do
    IFS=":" read -r varout varspec listtag <<< "${var_entry}"

    if [[ "${listtag}" == "plev" ]]; then
        datalist="${PLEV_LIST}"
    elif [[ "${listtag}" == "deriv" ]]; then
        datalist="${DERIV_LIST}"
    else
        datalist="${SFC_LIST}"
    fi

    out="${OUTDIR}/noc_min_msl_${varout}.nc"
    compose "${NOC_NODEFILE}" "${datalist}" "${varspec}" "${out}"
done

echo "==================== DONE ===================="
echo "Composites (OC copied, NOC matched 1:2) written to ${OUTDIR}"


