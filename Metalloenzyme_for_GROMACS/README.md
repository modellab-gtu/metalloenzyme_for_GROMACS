The current version requires following programs installed:
-----------
-----------
-Gaussian16
-AmberTools25 conda environment
-PyMol (no gui needed. only -cq mode will be used)

In order to use G16 efficiently on orfoz computers you have to run the following to modify the default G16 cores and memory:




sed -i 's/Mem=3000MB/Mem=220GB/g' $AMBERHOME/lib/python3.12/site-packages/pymsmt/mol/gauio.py
sed -i 's/NProcShared=2/NProcShared=110/g' $AMBERHOME/lib/python3.12/site-packages/pymsmt/mol/gauio.py

with your python version.

or simply run: run_scripts/fix_G16_default_for_orfoz.sh 

------------------------------------------------------------
----------------------------------------------------------

Required files are in run_scripts folder. There are two ways to model Zn protein 1) Bonded 2)Non-bonded

Bonded Approach:

MCPB.py is used along with G16 and tleap to model non standard residues. See details on https://ambermd.org/tutorials/advanced/tutorial20/mcpbpy.php and also Amber25.pdf manual

For bonded approach, just use run_scripts/prep_Zn_prot_byMCPBpy.sh file. In there you need to specify the 4 digit code of PDB id.

-------------------------------------------------------------------
-------------------------------------------------------------------

Non-bonded Approach:

Only tleap is used. The Zn ion is defined as LJ-1-------------------------------------------------------------------
-------------------------------------------------------------------

Non-bonded Approach:

Only tleap is used. The Zn ion is defined as LJ-12-6

run_scripts/prep_Zn_prot_nonbonded.sh is used.

