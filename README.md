# metalloenzyme_for_GROMACS

Preparing metalloproteins for molecular dynamics, with Amber parameters run in GROMACS or Amber.

| Folder | Contents |
|---|---|
| [`metalprep/`](metalprep/README.md) | **Automated pipeline.** Use this for new systems. It prepares any PDB entry, chain and metal with bonded models (MCPB.py + Gaussian 16, or ZAFF) or non-bonded models (Li/Merz 12-6 or 12-6-4). It writes Amber and GROMACS topologies, checks that their energies agree, and generates staged-restraint MD scripts and metal-site checks. |

The original manual Zn protocol (lysostaphin 4LXC), which metalprep automates, is in the git history: see commit [`6ccda33`](https://github.com/modellab-gtu/metalloenzyme_for_GROMACS/tree/6ccda330e4d60aee2907aafce7dcf22871043bbd/Metalloenzyme_for_GROMACS).

Related repositories:
- [modellab-gtu/charge_updated_MD](https://github.com/modellab-gtu/charge_updated_MD): MD with on-the-fly AIMNet2 charges.
- [modellab-gtu/useful_python_scripts](https://github.com/modellab-gtu/useful_python_scripts): alignment, RMSD clustering, G16 scan and geometry utilities.
