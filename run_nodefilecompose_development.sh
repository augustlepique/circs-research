#!/usr/bin/env bash

# run_nodefilecompose_development.sh
#
# Development-stage composites for the intensity and and season matched population
# (OC vs same month min msl matched NOC, 1:1). One nodefile per population per
# lead (min msl minus 0/6/12/18/24), written by TE_compositing_development.ipynb.
# ERA5 at every lead is already in era5_TE_composite(download_era5_composite.py
# OFFSET_HOURS), so the data lists are the same as run_nodefilecompose.sh.

set -euo pipefail

# SAMPLE selects the population: unset/"" = fixed sample (same 44 cyclones at every
# lead), "_varn" = every pair with a node at that lead (n varies with lead).
# Nodefiles and composites share the suffix, so they can't be mixed up.
SAMPLE="${SAMPLE:-}"
NODEDIR="/home1/lepique/circs_research/data/development_month1to1${SAMPLE}"
DATADIR="/data1/lepique/era5_TE_composite"
OUTDIR="/data1/lepique/composites_dev_month1to1${SAMPLE}"
mkdir -p "${OUTDIR}"

SFC_LIST="${DATADIR}/in_data_list_sfc.txt"
PLEV_LIST="${DATADIR}/in_data_list_plev.txt"
DERIV_LIST="${DATADIR}/in_data_list_derived.txt"
IRROT_LIST="${DATADIR}/in_data_list_irrot.txt"     # irrotational wind + PV advection (compute_irrotational.py)
PV_LIST="${DATADIR}/in_data_list_pv.txt"           # PV, all 21 levels (download_era5_composite.py --only pv)
W_LIST="${DATADIR}/in_data_list_w.txt"             # omega, all 21 levels (--only w)


DX=1.0
RESX=80
TIMEDELTA="1h"
INFMT="lon,lat,msl,phis"

LEADS=(24 18 12 6 0)
POPS=(oc noc)

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

for lead in "${LEADS[@]}"; do
    tag=$(printf "m%02dh" "${lead}")
    for pop in "${POPS[@]}"; do
        nodefile="${NODEDIR}/${pop}_min_msl_${tag}.txt"
        [[ -f "${nodefile}" ]] || { echo "!! missing ${nodefile}"; exit 1; }
        echo "==================== ${pop^^}  t-${lead}h  ($(grep -c '^start' "${nodefile}") cyclones) ===================="
        for var_entry in "${VARIABLES[@]}"; do
            IFS=":" read -r varout varspec listtag <<< "${var_entry}"
            case "${listtag}" in
                plev)  datalist="${PLEV_LIST}" ;;
                deriv) datalist="${DERIV_LIST}" ;;
                irrot) datalist="${IRROT_LIST}" ;;
                pv)    datalist="${PV_LIST}" ;;
                w)     datalist="${W_LIST}" ;;
                *)     datalist="${SFC_LIST}" ;;
            esac
            compose "${nodefile}" "${datalist}" "${varspec}" \
                    "${OUTDIR}/${pop}_${tag}_${varout}.nc" &
            while (( $(jobs -rp | wc -l) >= NJOBS )); do wait -n; done
        done
    done
done
wait

echo "==================== DONE ===================="
echo "Development-stage composites written to ${OUTDIR}"



