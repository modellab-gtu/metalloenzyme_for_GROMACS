#!/bin/bash
# metalprep on lysostaphin 4LXC chain B (Zn: His279 NE2, Asp283 OD1, His362 ND1; loop = PDB 391-406).
# Run on chn (needs internet for the PDB download) from an empty directory of your own.
set -e
module load metalprep

# 0) read the site report first (seconds)
metalprep prepare --pdb 4LXC --chain B --model mcpb --site-only -o 4LXC_site
cat 4LXC_site/site_report.txt

# 1) non-bonded 12-6 (no QM; minutes) -> GROMACS
metalprep prepare --pdb 4LXC --chain B --model nb126 --ion-set cm --loop 391-406 -o 4LXC_126

# 2) non-bonded 12-6-4 (no QM) -> Amber only
metalprep prepare --pdb 4LXC --chain B --model nb1264 --loop 391-406 -o 4LXC_1264

# 3) bonded MCPB: submits the Gaussian chain to cws02 (~2 h); 'metalprep finish' runs automatically
metalprep prepare --pdb 4LXC --chain B --model mcpb --loop 391-406 -o 4LXC_mcpb

# then, in any finished directory:
#   cd 4LXC_126 && bash submit_gromacs.sh        # staged EM + 10-replica array on cws01
#   cd 4LXC_1264 && sbatch run_amber.sbatch      # pmemd.cuda
#   metalprep check 4LXC_126 md6_0.gro           # Zn-donor distances vs crystal
