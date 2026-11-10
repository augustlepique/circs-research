#!/usr/bin/env bash

# run_nodefilecompose_matched_month.sh
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
OUTDIR="/data1/lepique/composites_matched_month1to1"
mkdir -p "${OUTDIR}"

NOC_NODEFILE="${NODEDIR}/nonoutbreak_cyclones_min_msl_cores_matched_month1to1.txt"
OC_NODEFILE="${NODEDIR}/outbreak_cyclones_min_msl_cores_filtered.txt"   # same OC population as composites_full_filtered

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
    # cp is not downloaded for era5_TE_composite yet (only the onset sfc files have it)
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


# data list for a variable-table LIST tag
datalist_for () {
    case "$1" in
        plev)  echo "${PLEV_LIST}" ;;
        deriv) echo "${DERIV_LIST}" ;;
        irrot) echo "${IRROT_LIST}" ;;
        pv)    echo "${PV_LIST}" ;;
        w)     echo "${W_LIST}" ;;
        *)     echo "${SFC_LIST}" ;;
    esac
}

echo "==================== OC: copy unchanged composites, compute new ones ===================="
for var_entry in "${VARIABLES[@]}"; do
    IFS=":" read -r varout varspec listtag <<< "${var_entry}"
    src="${SRCDIR}/oc_min_msl_${varout}.nc"
    dst="${OUTDIR}/oc_min_msl_${varout}.nc"
    if [[ -f "${dst}" ]]; then
        echo "  [skip] $(basename "${dst}") exists"
    elif [[ -f "${src}" ]]; then
        cp "${src}" "${dst}"
        echo "  [copy] $(basename "${src}")"
    else
        # not in composites_full_filtered (e.g. pv, w, the irrotational-wind
        # fields): compute it from the same OC population
        compose "${OC_NODEFILE}" "$(datalist_for "${listtag}")" "${varspec}" "${dst}"
    fi
done

echo "==================== NOC (matched 1:2) ===================="
for var_entry in "${VARIABLES[@]}"; do
    IFS=":" read -r varout varspec listtag <<< "${var_entry}"
    datalist="$(datalist_for "${listtag}")"

    out="${OUTDIR}/noc_min_msl_${varout}.nc"
    compose "${NOC_NODEFILE}" "${datalist}" "${varspec}" "${out}"
done

echo "==================== DONE ===================="
echo "Composites (OC copied, NOC matched 1:1, same month) written to ${OUTDIR}"

