#!/usr/bin/env bash
set -euo pipefail

pdbid=4LXC
catal_residues="r32 | r36 | r115"
loop_residues="r138-153"

wget https://files.rcsb.org/download/${pdbid}.pdb

grep "ATOM  " ${pdbid}.pdb > ${pdbid}_clean.pdb
grep "ZN"  ${pdbid}.pdb >>  ${pdbid}_clean.pdb
pymol -cq -d "load ${pdbid}_clean.pdb; select keep, chain A or byres (hetatm within 5 of chain A); save ${pdbid}_chainA_only.pdb, keep; quit"
pdb4amber -i ${pdbid}_chainA_only.pdb -o ${pdbid}_chainA_only_clean.pdb --reduce

tleap -f - <<EOF
source leaprc.protein.ff19SB
source leaprc.water.tip3p
loadamberparams frcmod.ions1lm_126_tip3p
mol = loadpdb ${pdbid}_chainA_only_clean.pdb
savepdb mol MOL_dry.pdb
saveamberparm mol MOL_dry.prmtop MOL_dry.inpcrd
solvatebox mol TIP3PBOX 10.0
addions mol Cl- 0
addions mol Na+ 0
savepdb mol MOL_solv.pdb
saveamberparm mol Zn_prot.prmtop Zn_prot.inpcrd
saveamberparm mol MOL_solv.prmtop MOL_solv.inpcrd
quit
EOF

## ---- 2) Convert AMBER -> GROMACS using ParmEd (inline python), with renaming ----
python - <<'PY'
from parmed import load_file

amber = load_file('MOL_solv.prmtop', 'MOL_solv.inpcrd')

# Normalize residue names:
# Waters: WAT/HOH -> SOL
# Ions:   Na+, Na, NA+ -> NA     ;  Cl-, Cl, CL- -> CL
for res in amber.residues:
    rn = res.name.strip()
    if rn in ('WAT', 'HOH'):
        res.name = 'SOL'
    elif rn in ('Na+', 'NA+', 'Na', 'NA'):
        res.name = 'NA'
    elif rn in ('Cl-', 'CL-', 'Cl', 'CL'):
        res.name = 'CL'
    # keep ZN as 'ZN' (divalent metal cofactor), do not rename

amber.save('Zn_prot.top', format='gromacs')
amber.save('Zn_prot.gro')
PY

# 1. Generate the initial index quietly
echo "q" | gmx make_ndx -f Zn_prot.gro -o index.ndx

# 2. Count how many groups exist and set the next index number
last=$(grep -c "^\[" index.ndx)
next=$((last + 0))

# 3. Now create new groups dynamically
gmx make_ndx -f Zn_prot.gro -o index.ndx -n index.ndx <<EOF
"Protein" | r ZN
name $next Protein_Zn
${catal_residues}
name  $((next+1)) catal
${loop_residues}
name $((next+2)) loop
"Protein-H" & "loop"
name $((next+3)) loop-H
"Protein-H" & !"loop-H" & !"catal"
name $((next+4)) Protein_H_nocatal_noloop
q
EOF


echo "Protein_H_nocatal_noloop" | gmx genrestr -f Zn_prot.gro -n index.ndx -o posre_Prot_H_nocatal_noloop.itp -fc 4000 4000 4000
echo "catal" | gmx genrestr -f Zn_prot.gro -n index.ndx -o posre_catal_H.itp -fc 5000 5000 5000
echo "loop-H" | gmx genrestr -f Zn_prot.gro -n index.ndx -o posre_loop_H.itp -fc 6000 6000 6000


awk '
FNR==1 {
  if      (FILENAME ~ /posre_Prot_H_nocatal_noloop\.itp$/) tag="fc_proH"
  else if (FILENAME ~ /posre_catal_H\.itp$/)               tag="fc_catalH"
  else if (FILENAME ~ /posre_loop_H\.itp$/)                tag="fc_loopH"
}
# lines that start with two integers
/^[[:space:]]*[0-9]+[[:space:]]+[0-9]+/ {
  i=$1; f=$2
  printf "%10d %5d %10s %10s %10s\n", i, f, tag, tag, tag
}' posre_Prot_H_nocatal_noloop.itp posre_catal_H.itp posre_loop_H.itp | sort -n -k1,1 > posre_body.tmp

cat > posre.itp <<'EOF'
#ifdef STEP4_0
#define fc_proH 4000
#define fc_catalH 4000
#define fc_loopH 4000
#endif

#ifdef STEP4_1
#define fc_proH 2000
#define fc_catalH 2000
#define fc_loopH 2000
#endif

#ifdef STEP4_2
#define fc_proH 1000
#define fc_catalH 1000
#define fc_loopH 1000
#endif

#ifdef STEP4_3
#define fc_proH 500
#define fc_catalH 500
#define fc_loopH 500
#endif

#ifdef STEP4_4
#define fc_proH 0
#define fc_catalH 0
#define fc_loopH 0
#endif
#ifdef CATAL
#define fc_catalH 4000
#endif
#ifdef LOOP
#define fc_loopH 4000
#endif
[ position_restraints ]
;    i funct        fcx        fcy        fcz
EOF

cat posre_body.tmp >> posre.itp

# 3d) Build posre_Zn.itp by copying header from posre.itp and adding a single-atom line with fc_mc
#     (Assumes Zn moleculetype has a single atom with local index 1.)
head -n 37 posre.itp > posre_Zn.itp
echo "         1     1      fc_catalH      fc_catalH      fc_catalH" >> posre_Zn.itp

# ---- 4) Patch the topology to include the restraint files at the END of each moleculetype ----
#     - first moleculetype  -> posre.itp
#     - moleculetype named ZN -> posre_Zn.itp
awk -v RS='\n' -v ORS='\n' '
BEGIN{
  inc1 = "#ifdef POSRES\n#include \"posre.itp\"\n#endif";
  inc2 = "#ifdef POSRES\n#include \"posre_Zn.itp\"\n#endif";
  in_mt=0; cur=""; mt_name_pending=0; first_mt=""; have1=have2=0;
}
function inject_if_needed(){
  if (cur != "") {
    if (cur == first_mt && !have1) print inc1;
    if (cur == "ZN"      && !have2) print inc2;
  }
  have1=have2=0; cur="";
}
{
  line=$0;

  # [ moleculetype ]
  if (match(line, /^[[:space:]]*\[[[:space:]]*moleculetype[[:space:]]*\][[:space:]]*$/)) {
    if (in_mt) inject_if_needed();     # close previous block with include
    in_mt=1; cur=""; mt_name_pending=1;
    print line; next;
  }

  # the name line after [ moleculetype ]
  if (mt_name_pending) {
    print line;
    if (line ~ /^[[:space:]]*$/ || line ~ /^[[:space:]]*;/) next;
    split(line, toks); cur=toks[0]?toks[0]:toks[1];  # first token
    if (first_mt == "") first_mt = cur;
    have1=have2=0; mt_name_pending=0;
    next;
  }

  # detect pre-existing includes to avoid duplicates
  if (in_mt && cur != "") {
    if (line ~ /[[:space:]]posre\.itp[[:space:]]*$/)    have1=1;
    if (line ~ /[[:space:]]posre_Zn\.itp[[:space:]]*$/) have2=1;
  }

  print line;
}
END{
  if (in_mt) inject_if_needed();       # close last block at EOF
}
' Zn_prot.top > Zn_prot.top.patched && mv Zn_prot.top.patched Zn_prot.top

echo "All done:
  - AMBER: Zn_prot.prmtop / Zn_prot.inpcrd
  - GROMACS: Zn_prot.top / Zn_prot.gro
  - Restraints: posre.itp / posre_Zn.itp
  - Topology patched to include restraints at the end of each moleculetype block."

