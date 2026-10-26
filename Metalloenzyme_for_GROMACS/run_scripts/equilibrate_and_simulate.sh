#!/usr/bin/env bash
set -euo pipefail

export GMX_MAXBACKUP=-1

module load apps/gromacs/2024.1-oneapi2024

TOP=Zn_prot.top
NDX=index.ndx
START_GRO=Zn_prot.gro

for rep in {0..9};do

## --- Initial NVT equilibration (no stepwise macros) ---
gmx_mpi grompp -f MDP/nvt50.mdp -n "$NDX" -c em_cg44.gro -r em_cg44.gro -p "$TOP" -o nvt50_${rep}.tpr
mpirun -np 110 gmx_mpi mdrun -deffnm nvt50_${rep}

# --- Stepwise release NVT: 4.1 → 4.4 ---
for i in {1..4}; do
  j=$((i-1))
  echo ">>> STEP 4.$i  (previous = 4.$j)"

  # Steepest descent at STEP4_i using last CG structure (4.j)
  gmx_mpi grompp -f "MDP/nvt5${i}.mdp" -n "$NDX" -c "nvt5${j}_${rep}.gro"  -r "nvt5${j}_${rep}.gro"  -p "$TOP" -o "nvt5${i}_${rep}.tpr"
  mpirun -np 110 gmx_mpi mdrun -deffnm "nvt5${i}_${rep}"

done
## NPT equilibration
gmx_mpi grompp -f "MDP/npt5.mdp" -n "$NDX" -c "nvt54_${rep}.gro" -r "nvt54_${rep}.gro" -p "$TOP" -o "npt5_${rep}.tpr"
mpirun -np 110 gmx_mpi mdrun -deffnm "npt5_${rep}"
  

# Final MD equilibration
gmx_mpi grompp -f "MDP/md6.mdp" -n "$NDX" -c "npt5_${rep}.gro" -r "npt5_${rep}.gro" -p "$TOP" -o "md6_${rep}.tpr"
mpirun -np 110 gmx_mpi mdrun -deffnm "md6_${rep}"

done
