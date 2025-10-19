#!/usr/bin/env bash
set -euo pipefail

export GMX_MAXBACKUP=-1
alias gmx="gmx_mpi"

mdfile=md6
trjfile="trjnojump_prozn_highres"

for rep in {0..0}; do
  for wrt in xraystr startstr; do

    if [[ "$wrt" == "xraystr" ]]; then
      strfile="em_steep40"
    elif [[ "$wrt" == "startstr" ]]; then
      strfile="${mdfile}_${rep}"
    fi


    # 1. Remove jumps
    printf "17\n" | gmx trjconv -f ${mdfile}_${rep}.xtc -s em_steep40.tpr -pbc nojump -ur compact -o ${trjfile}nofit_${rep}.xtc -n index.ndx
    
    # 2. Fit to EM structure (same tpr)
    printf "4\n17\n" |gmx trjconv -f ${trjfile}nofit_${rep}.xtc -s em_steep40.tpr -fit rot+trans -o ${trjfile}_${rep}.xtc -n index.ndx
    rm ${trjfile}nofit_${rep}.xtc

    # Gyration
    printf "17\n" | gmx gyrate -f "${trjfile}_${rep}.xtc" -s "${strfile}.tpr" -o "gyr_prozn_${wrt}_${rep}.xvg" -n index.ndx
    printf "4\n"  | gmx gyrate -f "${trjfile}_${rep}.xtc" -s "${strfile}.tpr" -o "gyr_back_${wrt}_${rep}.xvg"  -n index.ndx

    # RMSF (per-residue and per-atom) + average structure
    printf "17\n" | gmx rmsf -f "${trjfile}_${rep}.xtc" -s "${strfile}.tpr" -o "rmsf-per-residue_${wrt}_${rep}.xvg" \
                           -ox "average_${wrt}_${rep}.pdb" -oq "bfactors-residue_${wrt}_${rep}.pdb" -res -n index.ndx
    printf "17\n" | gmx rmsf -f "${trjfile}_${rep}.xtc" -s "${strfile}.tpr" -o "rmsf-per-atom_${wrt}_${rep}.xvg" \
                           -oq "bfactors-atom_${wrt}_${rep}.pdb" -n index.ndx

    # RMSD vs reference (whole traj)
    printf "17\n17\n" | gmx rms -f "${trjfile}_${rep}.xtc" -s "${strfile}.tpr" -o "rmsd-all-atom_${wrt}_${rep}.xvg" -n index.ndx
    printf "1\n1\n"   | gmx rms -f "${trjfile}_${rep}.xtc" -s "${strfile}.tpr" -o "rmsd-prot_${wrt}_${rep}.xvg"      -n index.ndx
    printf "4\n4\n"   | gmx rms -f "${trjfile}_${rep}.xtc" -s "${strfile}.tpr" -o "rmsd-back_${wrt}_${rep}.xvg"      -n index.ndx


  done
done

