#!/usr/bin/env bash
set -euo pipefail

export GMX_MAXBACKUP=-1

module load apps/gromacs/2024.1-oneapi2024

TOP=Zn_prot.top
NDX=index.ndx
START_GRO=Zn_prot.gro

# --- Initial minimization (no stepwise macros) ---
gmx_mpi grompp -f MDP/em_steep.mdp -n "$NDX" -c "$START_GRO"  -r "$START_GRO"  -p "$TOP" -o em_steep.tpr
mpirun -np 110 gmx_mpi mdrun -v -deffnm em_steep

gmx_mpi grompp -f MDP/em_cg.mdp    -n "$NDX" -c em_steep.gro   -r em_steep.gro   -p "$TOP" -o em_cg.tpr
mpirun -np 110 gmx_mpi mdrun -v -deffnm em_cg

# --- Step 4.0 (strongest restraints) ---
gmx_mpi grompp -f MDP/em_steep40.mdp -n "$NDX" -c em_cg.gro      -r em_cg.gro      -p "$TOP" -o em_steep40.tpr
mpirun -np 110 gmx_mpi mdrun -v -deffnm em_steep40

gmx_mpi grompp -f MDP/em_cg40.mdp    -n "$NDX" -c em_steep40.gro -r em_steep40.gro -p "$TOP" -o em_cg40.tpr
mpirun -np 110 gmx_mpi mdrun -v -deffnm em_cg40

# --- Stepwise release: 4.1 → 4.4 ---
for i in {1..4}; do
  j=$((i-1))
  echo ">>> STEP 4.$i  (previous = 4.$j)"

  # Steepest descent at STEP4_i using last CG structure (4.j)
  gmx_mpi grompp -f "MDP/em_steep4${i}.mdp" -n "$NDX" -c "em_cg4${j}.gro"  -r "em_cg4${j}.gro"  -p "$TOP" -o "em_steep4${i}.tpr"
  mpirun -np 110 gmx_mpi mdrun -v -deffnm "em_steep4${i}"

  # Conjugate gradient at STEP4_i using the SD output of the same i
  gmx_mpi grompp -f "MDP/em_cg4${i}.mdp"    -n "$NDX" -c "em_steep4${i}.gro" -r "em_steep4${i}.gro" -p "$TOP" -o "em_cg4${i}.tpr"
  mpirun -np 110 gmx_mpi mdrun -v -deffnm "em_cg4${i}"
done

