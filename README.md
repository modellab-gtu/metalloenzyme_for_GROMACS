# metalloenzyme_for_GROMACS: metalprep

Automated metalloprotein preparation for MD, with Amber parameters run in GROMACS or Amber.

| Path | Contents |
|---|---|
| `bin/`, `src/`, `examples/`, `patches/` | **metalprep**, the automated pipeline, described below. |
| [`Metalloenzyme_for_GROMACS/`](Metalloenzyme_for_GROMACS/README.md) | The original manual Zn protocol (lysostaphin 4LXC, mostly bash scripts), which metalprep automates. |

Related repositories:
- [modellab-gtu/charge_updated_MD](https://github.com/modellab-gtu/charge_updated_MD): MD with on-the-fly AIMNet2 charges.
- [modellab-gtu/useful_python_scripts](https://github.com/modellab-gtu/useful_python_scripts): alignment, RMSD clustering, G16 scan and geometry utilities.

```
module load metalprep            # also loads md/ambertools-25 and gromacs/2026.1-tmpi
metalprep prepare --pdb 4LXC --chain B --model nb1264 -o 4LXC_1264
```

metalprep automates the GTU Zn-enzyme protocol (modellab-gtu/metalloenzyme_for_GROMACS, first tested on lysostaphin
4LXC) for any PDB entry, chain and metal site. You choose the metal model:

| `--model` | Metal model | QM? | Engines | When to use |
|---|---|---|---|---|
| `mcpb` | bonded, MCPB.py + Gaussian 16 (Seminario force constants, RESP charges) | yes, about 1–3 h on cws02 | GROMACS, Amber | any site, the most specific model |
| `zaff` | bonded, ZAFF library (Peters et al. 2010) | no | GROMACS, Amber | 4-coordinate Zn sites that match a ZAFF centre (Cys/His/Asp/water combinations) |
| `nb126` | non-bonded Li/Merz 12-6 ion (`--ion-set cm \| hfe \| iod`) | no | GROMACS, Amber | quick setups; the Zn often drifts away from its crystal ligands |
| `nb1264` | non-bonded Li/Merz 12-6-4 ion | no | **Amber / OpenMM only** | keeps the coordination without QM; GROMACS cannot run 12-6-4 |

## What it does
1. **Input.** It downloads the PDB entry, or reads a local file. It keeps the chain(s) you name with `--chain`, their metal
   ions (`--metal ZN:2`, repeatable) and any crystal waters bound to the metal.
   - Other hetero groups that bind the metal (for example a crystallisation sulfate) are removed, with a warning.
2. **Protonation.** It runs `pdb4amber --reduce`, then analyses each metal site using donor atoms within `--cutoff` (2.8 Å).
   - Each ligating His becomes **HID or HIE according to which nitrogen binds the metal**. Cys becomes CYM.
   - Override a choice with `--his 279:HIE`, using the PDB numbering.
   - Everything is listed in `site_report.txt`; **read it before the QM step.**
3. **Parameters.** These depend on `--model`.
   - With `mcpb`, MCPB.py step 1 runs at once. The three Gaussian jobs then run as a Slurm chain on cws02:
     - small-model optimisation, then force constants;
     - large-model ESP for the charges, in parallel with the first two.

     `metalprep finish` follows automatically: MCPB steps 2–4, then the system build.
   - Gaussian resources: `--qm-cores 32 --qm-mem 50GB`, the defaults, fit two jobs side by side on cws02.
4. **System.** It solvates the protein (`--water tip3p`, `--box 10` Å, `--box-shape box|oct`), neutralises it and adds
   NaCl (`--salt 0.15` M), writing `system.prmtop/inpcrd`.
   - For every model except 12-6-4, it also writes `system.top/gro`, converted with ParmEd.
5. **Checks and run files.**
   - **Energy check:** the Amber and GROMACS topologies must give the same solute energy, term by term (`energy_check.txt`).
   - **Index groups:** `index.ndx` has named groups, including `catal` (the metal plus its ligating residues), `loop`
     (`--loop 391-406` in PDB numbering, or `--loop new:138-153` for the renumbered system) and
     `Protein_H_nocatal_noloop`.
   - **Restraints:** staged position restraints `posre_*.itp` follow the original protocol. The force constants are 4000,
     2000, 1000, 500 and 0, selected with `-DSTEP4_0 … -DSTEP4_4`; `-DCATAL` and `-DLOOP` hold those groups.
   - **GROMACS run files:** the protocol's `MDP/` files and `submit_gromacs.sh`, which runs a staged minimisation and
     then a replica array (`--replicas 10`). Each replica does an NVT release, NPT and 20 ns of production. GPU (cws01)
     is the default; `--md-node cws02` runs on CPU.
   - **Amber run file:** `run_amber.sbatch` (pmemd.cuda), the route to use for 12-6-4.
   - **Analysis:** `md_analysis.sh`.

## Commands
A worked example script is in `examples/4LXC.sh`.

```
metalprep prepare --pdb 4LXC --chain B --model mcpb   -o 4LXC_mcpb    # QM chain submitted; wait for mp_finish
metalprep prepare --pdb 2GIV.pdb       --model zaff   -o 2GIV_zaff
metalprep prepare --pdb 4LXC --chain B --model nb126  --ion-set iod -o 4LXC_126
metalprep prepare --pdb 4LXC --chain B --model nb1264 -o 4LXC_1264
metalprep prepare ... --site-only        # only the site report (check His states before spending QM time)
cd 4LXC_mcpb && bash submit_gromacs.sh   # or: sbatch run_amber.sbatch
metalprep check 4LXC_mcpb md6_0.gro      # metal–donor distances vs the crystal (also *.rst7, or *.xtc with --tpr)
```
`metalprep check` flags any metal–donor distance that has moved more than 0.5 Å from the crystal. The minimisation and MD
scripts run it automatically.

## Validation (GTU)
- **MCPB step 1 (1A5T):** the AmberTools regression test gives all 28 files identical to the reference.
- **MCPB full chain (4LXC chain B):** Gaussian opt (30 min), fc (8 min) and ESP (84 min) on 32 cores of cws02, then
  `finish`. Zn bonds: His NE2 1.98 Å (k 92.8), Asp OD1 2.02 Å (k 49.9), His ND1 1.98 Å (k 92.1); Zn RESP charge +1.09.
  Amber and GROMACS energies are identical term by term. After minimisation and 150 ps NVT/NPT/MD on cws01
  (about 230 ns/day), every Zn–donor distance is within 0.25 Å of the crystal.
- **ZAFF (2GIV):** metalprep finds the same centre as the Amber ZAFF tutorial (Cys₃His with HID, centre 3) and
  writes the same four Zn bonds.
- **Amber ↔ GROMACS:** the solute energies agree term by term within 0.002 kJ/mol for both 4LXC (12-6) and 2GIV (ZAFF).
- **ff19SB CMAP fix:** plain ParmEd conversion merges ff19SB's 13 residue-specific CMAP corrections into one. On 4LXC
  that gives a CMAP energy of 668 instead of 948 kJ/mol. metalprep keeps the classes separate.
- **Short GPU runs:** the full GROMACS chain (staged EM, NVT release, NPT, MD) and the Amber chain both run.
- **4LXC chain B site:** His279 binds through NE2 (HID), His362 through ND1 (HIE), and Asp283 through OD1. The fourth
  position is open: it holds a crystallisation sulfate in chain A and nothing in chains B–D.

## Installation
**GTU ModelLab cluster:** `module load metalprep`. It is already installed in `/cta/capps/envs/AmberTools25/opt/metalprep`.

**Elsewhere:**
1. Requirements:
   - an AmberTools 25 conda environment (provides tleap, pdb4amber, MCPB.py, parmed and openmm);
   - GROMACS 2023 or newer;
   - Gaussian 16, only for `--model mcpb`.
2. Point the launcher at that environment: `export METALPREP_AMBERTOOLS=/path/to/AmberTools25`.
3. For `mcpb` only: apply `patches/pymsmt_gauio_mem_env.patch` to the environment's `site-packages`
   (`cd $METALPREP_AMBERTOOLS/lib/python3.*/site-packages && patch -p1 < .../pymsmt_gauio_mem_env.patch`).
   Stock pymsmt hard-codes `%Mem=3000MB` and `%NProcShared=2`; the patch makes it read `MCPB_GAU_MEM` and `MCPB_GAU_NPROC`,
   keeping those values as defaults. It is already applied on the GTU cluster.
4. Run `src/metalprep_data/get_zaff.sh` once to download the ZAFF library from ambermd.org.
5. Put `bin/` on your PATH.

The generated Slurm scripts use GTU defaults (partition `main`; QM on `cws02`; MD on `cws01` with GPU or `cws02` with CPU; `gaussian/g16-avx2` and `gromacs/2026.1-tmpi(-cuda)` modules), so adapt them for other clusters.

The original manual scripts for the 4LXC protocol are in [`Metalloenzyme_for_GROMACS/`](Metalloenzyme_for_GROMACS/README.md). **Do not run its `fix_G16_default_for_orfoz.sh`**: metalprep sets the Gaussian memory and cores through the `MCPB_GAU_MEM` and `MCPB_GAU_NPROC` environment variables instead of editing the shared pymsmt files.

## Notes
- Protein force field: `--ff ff19SB` (default) or `ff14SB`. ZAFF and MCPB residues are ff14SB-era parameters. The
  ZAFF tutorial uses ff14SB, which may be the safer choice for `zaff`.
- **Open-shell metals** (Fe, Cu, Mn): give the spin with `--mult`. Only Zn has been validated so far.
- **ZAFF does not cover** 5- or 6-coordinate Zn, or sites with two bound waters (centres 11 and 12); use `mcpb`.
- **Gaussian** is required only for `mcpb`, through `gaussian/g16-avx2`. MCPB.py supports Gaussian and GAMESS, not ORCA.
- Gaussian `%Mem` and `%NProcShared` come from `MCPB_GAU_MEM` and `MCPB_GAU_NPROC`, which metalprep sets from
  `--qm-mem` and `--qm-cores`. **Never edit the shared pymsmt files.**
- References: MCPB.py, Li & Merz, *J. Chem. Inf. Model.* 56, 599 (2016). ZAFF, Peters et al., *J. Chem. Theory Comput.* 6,
  2935 (2010). 12-6 and 12-6-4 ions, Li & Merz, *J. Chem. Theory Comput.* 10, 289 (2014), and Li, Song & Merz,
  *J. Chem. Theory Comput.* 11, 1645 (2015).
