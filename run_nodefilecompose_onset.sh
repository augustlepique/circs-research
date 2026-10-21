#!/usr/bin/env bash

# run_nodefilecompose_onset.sh
#
# Outbreak-cyclone composites lagged relative to the first KDE core's onset and
# peak time (ref minus 0/6/12/18/24 h). One nodefile per reference per lag,
# written by the onset/peak section of TE_compositing_development.ipynb. ERA5 for
# every needed hour is in era5_TE_composite_onset (python_scripts/
# download_era5_onset.py), on the expanded AREA [80, -180, -5, -20] so the +/-25
# deg plotted window is fully covered for ~96% of centers. Only that directory's
# data lists are used, so no composite mixes the old and new domains.

set -euo pipefail

NODEDIR="/home1/lepique/circs_research/data/onset_peak_oc"
DATADIR="/data1/lepique/era5_TE_composite_onset"
OUTDIR="/data1/lepique/composites_onset_peak_oc"
mkdir -p "${OUTDIR}"

SFC_LIST="${DATADIR}/in_data_list_sfc.txt"
PLEV_LIST="${DATADIR}/in_data_list_plev.txt"
DERIV_LIST="${DATADIR}/in_data_list_derived.txt"

DX=1.0
RESX=80
TIMEDELTA="1h"
INFMT="lon,lat,msl,phis"

LAGS=(24 18 12 6 0)
REFS=(onset peak)

NJOBS=${NJOBS:-6}        # concurrent NodeFileCompose runs

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
    # terminal output of NodeFileCompose is written to ${out%.nc}.log
    NodeFileCompose \
        --in_nodefile      "${nodefile}" \
        --in_nodefile_type "SN" \
        --in_fmt           "${INFMT}" \
        --in_data_list     "${datalist}" \
        --out_data         "${out}.tmp" \
        --var              "${varspec}" \
        --op               "mean" \
        --snapshots \
        --max_time_delta   "${TIMEDELTA}" \
        --dx               "${DX}" \
        --resx             "${RESX}" \
        --lonname          "longitude" \
        --latname          "latitude" \
        --regional > "${out%.nc}.log" 2>&1 \
    && mv "${out}.tmp" "${out}" \
    || echo "  !! FAILED: $(basename "${out}") (see ${out%.nc}.log)"
}

for lag in "${LAGS[@]}"; do
    tag=$(printf "m%02dh" "${lag}")
    for ref in "${REFS[@]}"; do
        nodefile="${NODEDIR}/oc_${ref}_${tag}.txt"
        [[ -f "${nodefile}" ]] || { echo "!! missing ${nodefile}"; exit 1; }
        echo "==================== OC  ${ref}-${lag}h  ($(grep -c '^start' "${nodefile}") cyclones) ===================="
        for var_entry in "${VARIABLES[@]}"; do
            IFS=":" read -r varout varspec listtag <<< "${var_entry}"
            case "${listtag}" in
                plev)  datalist="${PLEV_LIST}" ;;
                deriv) datalist="${DERIV_LIST}" ;;
                *)     datalist="${SFC_LIST}" ;;
            esac
            compose "${nodefile}" "${datalist}" "${varspec}" \
                    "${OUTDIR}/oc_${ref}_${tag}_${varout}.nc" &
            while (( $(jobs -rp | wc -l) >= NJOBS )); do wait -n; done
        done
    done
done
wait

echo "==================== DONE ===================="
echo "Onset/peak-lagged OC composites written to ${OUTDIR}"
