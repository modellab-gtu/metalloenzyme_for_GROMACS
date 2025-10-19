#!/bin/bash
#SBATCH -p orfoz
#SBATCH -A akocak
#SBATCH -J LSS
#SBATCH -N 1
#SBATCH -n 110
#SBATCH -C weka
#SBATCH --time=3-00:00:00

module load apps/gromacs/2024.1-oneapi2024



bash run_scritps/equilibrate_and_simulate.sh

