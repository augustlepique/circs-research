#!/bin/bash -l

############ USER OPTIONS #####################
## If using unstructured data, need connect file, otherwise empty
CONNECTFLAG=""
DO_TRACKS=true
DO_EXTRACT=false
ENSEMBLEMEMBER=r1i1p1f1   #r1, r2, etc.

## Unique string
#UQSTR="GFDL-CM4"
#PARENTSTR="NOAA-GFDL"
#GRIDDATE="gr1/v20180701"
#STYR="-1"

UQSTR="ERA5"
STYR="1980"

############ MACHINE SPECIFIC AUTO-CONFIG #####################

if [[ $HOSTNAME = cheyenne* ]]; then
  TEMPESTEXTREMESDIR=/glade/u/home/zarzycki/work/tempestextremes_noMPI/
  PATHTOFILES=/glade/collections/cmip/CMIP6/CMIP/${PARENTSTR}/${UQSTR}/historical/${ENSEMBLEMEMBER}/6hrPlevPt/psl/${GRIDDATE}/psl/
  PATHTOFILES2=/glade/collections/cmip/CMIP6/CMIP/${PARENTSTR}/${UQSTR}/historical/${ENSEMBLEMEMBER}/3hr/pr/${GRIDDATE}/pr/
  PATHTOFILES3=/glade/scratch/zarzycki/CMIPTMP/${UQSTR}/
  TOPOFILE=/glade/work/zarzycki/topo-files/CMIP6/${UQSTR}.topo.nc
  THISSED="sed"
elif [[ $HOSTNAME = derecho* ]]; then
  TEMPESTEXTREMESDIR=/glade/u/home/zarzycki/work/tempestextremes_noMPI/
  PATHTOFILES=/glade/campaign/collections/rda/data/d633000/e5.oper.an.sfc/
  PATHTOFILES2=/glade/campaign/collections/rda/data/d633000/e5.oper.fc.sfc.meanflux/
  PATHTOFILES3=/glade/u/home/mgore/scratch/CMIPTMP/${UQSTR}/
  TOPOFILE=/glade/u/home/mgore/work/topo-files/${UQSTR}.topo.nc
  TOPOORIGDIR=/glade/work/zarzycki/cam_tools/hires-topo/
  THISSED="sed"
elif [[ $(hostname -s) = MET-MAC* ]]; then
  TEMPESTEXTREMESDIR=/Users/cmz5202/Software/tempestextremes/
  PATHTOFILES=/Users/cmz5202/NetCDF/CMIP6/
  TOPOFILE=/Users/cmz5202/NetCDF/topo_files/${UQSTR}.topo.nc
  THISSED="gsed"
elif [[ $(hostname -s) = aci* ]]; then
  TEMPESTEXTREMESDIR=/storage/home/cmz5202/sw/tempestextremes/
  PATHTOFILES=/gpfs/group/cmz5202/default/mjg6459/HighResMIP/${PARENTSTR}/${UQSTR}/
  PATHTOFILES3=/storage/home/${LOGNAME}/scratch/CMIPTMP/${UQSTR}/
  TOPOFILE=/storage/home/mjg6459/work/topo/${UQSTR}.topo.nc
#  TOPOFILE=/gpfs/group/cmz5202/default/topo/${UQSTR}.topo.nc
  TOPOORIGDIR=/storage/home/cmz5202/work/cam_tools/hires-topo/
  THISSED="sed"
elif [[ $(hostname -s) = submit* ]]; then
  TEMPESTEXTREMESDIR=/storage/home/cmz5202/sw/tempestextremes/
  PATHTOFILES=/gpfs/group/cmz5202/default/mjg6459/HighResMIP/${PARENTSTR}/${UQSTR}/
  PATHTOFILES3=/storage/home/${LOGNAME}/scratch/CMIPTMP/${UQSTR}/
  TOPOFILE=/storage/home/mjg6459/work/topo/${UQSTR}.topo.nc
#  TOPOFILE=/gpfs/group/cmz5202/default/topo/${UQSTR}.topo.nc
  TOPOORIGDIR=/storage/home/cmz5202/work/cam_tools/hires-topo/
  THISSED="sed"
else
  echo "Can't figure out hostname, exiting"
  exit
fi

############ TRACKER MECHANICS #####################
starttime=$(date -u +"%s")

DATESTRING=`date +"%s%N"`
OUTTRAJDIR="./trajs/"
TRAJFILENAME=${OUTTRAJDIR}/trajsens2.txt.${UQSTR}
FILELISTNAME=filelist.txt.${DATESTRING}
NODELISTNAME=nodelist.txt.${DATESTRING}
NODELISTNAMEFILT=nodelistfilt.txt.${DATESTRING}

########### SET UP FOLDERS #####################

mkdir -p ${PATHTOFILES3}
mkdir -p $(dirname "${TOPOFILE}")

############ TRACK ETCs #####################

## Build trajectory files; the most time intensive part of this.
if [ "$DO_TRACKS" = true ] ; then

  mkdir -p $OUTTRAJDIR

  touch $FILELISTNAME

  #### Files
  find ${PATHTOFILES} -name "e5.oper.an.sfc.128_151_msl.*nc" | grep .1980 | grep -v .1981 | sort -n >> $FILELISTNAME
#  find ${PATHTOFILES} -name "psl_6hrPlevPt_${UQSTR}_hist*_*nc" | sort -n >> $FILELISTNAME
#  find ${PATHTOFILES} -name "E3SMHRHR.h1.*nc" | sort -n >> $FILELISTNAME
  # Add any static file(s) to each line
  $THISSED -e 's?$?;'"${TOPOFILE}"'?' -i $FILELISTNAME
  cat $FILELISTNAME

  ## Check if topo file exists, if not generate one using the first PSL file
  if [ ! -f ${TOPOFILE} ]; then
    echo "Topo file ${TOPOFILE} not found!"
    FIRSTPSLFILE=$(echo $(head -n 1 ${FILELISTNAME}) | awk -F';' '{print $1}')
    export TOPOORIGDIR="${TOPOORIGDIR}"  # Send topo dir to shell var for use inside NCL
    ncl data-process/make_topo_file.ncl 'trackerFileName="'$FIRSTPSLFILE'"' 'outFileName="'${TOPOFILE}'"'
  else
    echo "Topo file ${TOPOFILE} already exists."
  fi

#  Original
#  DCU_PSLNAME="psl"    # Name of PSL on netcdf files
#  DCU_PSLFOMAG=200.0   # Pressure "anomaly" required for PSL min to be kept
#  DCU_PSLFODIST=6.0    # Pressure "anomaly" contour must lie within this distance
#  DCU_MERGEDIST=6.0    # Merge two competing PSL minima within this radius
#  SN_TRAERA5NGE=6.0     # Maximum distance (GC degrees) ETC can move in successive steps
#  SN_TRAJMINLENGTH=10  # Min length of trajectory (nsteps)
#  SN_TRAJMAXGAP=3      # Max gap within a trajectory (nsteps)

#  Sensitivity 1
#  DCU_PSLNAME="psl"    # Name of PSL on netcdf files
#  DCU_PSLFOMAG=200.0   # Pressure "anomaly" required for PSL min to be kept
#  DCU_PSLFODIST=6.0    # Pressure "anomaly" contour must lie within this distance
#  DCU_MERGEDIST=6.0    # Merge two competing PSL minima within this radius
#  SN_TRAJRANGE=6.0     # Maximum distance (GC degrees) ETC can move in successive steps
#  SN_TRAJMINLENGTH=4   # Min length of trajectory (nsteps)
#  SN_TRAJMAXGAP=1      # Max gap within a trajectory (nsteps)
#  SN_MINENDPOINT=9.01  # Min travel distance
#  SN_MINTIME="24h" 

#  Sensitivity 2 -- 6-HOURLY
#  DCU_PSLNAME="MSL"    # Name of PSL on netcdf files
#  DCU_PSLFOMAG=200.0   # Pressure "anomaly" required for PSL min to be kept
#  DCU_PSLFODIST=6.0    # Pressure "anomaly" contour must lie within this distance
#  DCU_MERGEDIST=9.01   # Merge two competing PSL minima within this radius
#  SN_TRAJRANGE=9.01   # Maximum distance (GC degrees) ETC can move in successive steps
#  SN_TRAJMINLENGTH=4  # Min length of trajectory (nsteps)
#  SN_TRAJMAXGAP=1      # Max gap within a trajectory (nsteps)
#  SN_MINENDPOINT=12.0  # Min travel distance
#  SN_MINTIME="24h"

#  STRDETECT="--verbosity 0 --timestride 6 ${CONNECTFLAG} --out cyclones_tempest.${DATESTRING} --closedcontourcmd ${DCU_PSLNAME},${DCU_PSLFOMAG},${DCU_PSLFODIST},0 --mergedist ${DCU_MERGEDIST} --searchbymin ${DCU_PSLNAME} --outputcmd ${DCU_PSLNAME},min,0;PHIS,max,0 --latname "latitude" --lonname "longitude""

#  Sensitivity 2 -- HOURLY
  DCU_PSLNAME="MSL"    # Name of PSL on netcdf files
  DCU_PSLFOMAG=200.0   # Pressure "anomaly" required for PSL min to be kept
  DCU_PSLFODIST=6.0    # Pressure "anomaly" contour must lie within this distance
  DCU_MERGEDIST=9.01   # Merge two competing PSL minima within this radius
  SN_TRAJRANGE=9.01   # Maximum distance (GC degrees) ETC can move in successive steps
  SN_TRAJMINLENGTH=24  # Min length of trajectory (nsteps)
  SN_TRAJMAXGAP=6      # Max gap within a trajectory (nsteps)
  SN_MINENDPOINT=12.0  # Min travel distance
  SN_MINTIME="24h"

  STRDETECT="--verbosity 0 --timestride 1 ${CONNECTFLAG} --out cyclones_tempest.${DATESTRING} --closedcontourcmd ${DCU_PSLNAME},${DCU_PSLFOMAG},${DCU_PSLFODIST},0 --mergedist ${DCU_MERGEDIST} --searchbymin ${DCU_PSLNAME} --outputcmd ${DCU_PSLNAME},min,0;PHIS,max,0 --latname "latitude" --lonname "longitude""
  echo $STRDETECT
  touch cyclones.${DATESTRING}
  ${TEMPESTEXTREMESDIR}/bin/DetectNodes --in_data_list "${FILELISTNAME}" ${STRDETECT} </dev/null
  cat cyclones_tempest.${DATESTRING}* >> cyclones.${DATESTRING}
  rm cyclones_tempest.${DATESTRING}*

  ### Special filtering by time to account for fact that some models have longer PSL datasets than PRECT or PRECSN
  ## Create a temp cyclones file only including dates from STYR onwards, then overwrite full cyclones file before SN
  if [ $STYR \> 0 ]; then
    ${THISSED} -n '/^'${STYR}'/,$p' cyclones.${DATESTRING} > cycfilter.tmp.${DATESTRING}.txt;
    mv cycfilter.tmp.${DATESTRING}.txt cyclones.${DATESTRING}
  fi;

  # Stitch candidate cyclones together
#  STRSTITCH="--range ${SN_TRAERA5NGE} --mintime 60h --minlength ${SN_TRAJMINLENGTH} --maxgap ${SN_TRAJMAXGAP} --in cyclones.${DATESTRING} --out ${TRAJFILENAME} --min_endpoint_dist 12.0 --threshold lat,>,24,1;lon,>,234,1;lat,<,52,1;lon,<,294,1"
  STRSTITCH="--range ${SN_TRAJRANGE} --mintime ${SN_MINTIME} --minlength ${SN_TRAJMINLENGTH} --maxgap ${SN_TRAJMAXGAP} --in cyclones.${DATESTRING} --out ${TRAJFILENAME} --min_endpoint_dist ${SN_MINENDPOINT} --threshold lat,>,24,1;lon,>,234,1;lat,<,52,1;lon,<,294,1"
  ${TEMPESTEXTREMESDIR}/bin/StitchNodes --in_fmt "lon,lat,slp,phis" ${STRSTITCH}

  # Clean up leftover files
  rm cyclones.${DATESTRING}   #Delete candidate cyclone file
  rm ${FILELISTNAME}
  rm log*.txt

  ${TEMPESTEXTREMESDIR}/bin/HistogramNodes --in ${TRAJFILENAME} --nlat 36 --nlon 72 --iloncol 3 --ilatcol 4 --out "${PATHTOFILES3}/density_${UQSTR}.nc"
fi

endtime=$(date -u +"%s")
tottime=$(($endtime-$starttime))
printf "${tottime},${TRAJFILENAME}\n" >> timing.txt

