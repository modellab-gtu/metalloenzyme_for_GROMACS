#!/usr/bin/env python3
"""metalprep: automated metalloprotein preparation for MD (GTU ModelLab).

Flow (generalised from modellab-gtu/metalloenzyme_for_GROMACS, 4LXC Zn test case):
  1. fetch/read PDB -> keep chosen chains + metal ions (+ metal-bound waters)
  2. pdb4amber --reduce -> metal-site analysis -> ligating His set to HID/HIE from the bound N,
     Cys -> CYM; site report
  3. parameters:  mcpb   bonded, MCPB.py + Gaussian 16 (QM runs as a Slurm dependency chain)
                  zaff   bonded, ZAFF library (4-coordinate Zn centres only, no QM)
                  nb126  non-bonded Li/Merz 12-6 ion (hfe | iod | cm set)
                  nb1264 non-bonded Li/Merz 12-6-4 ion (Amber/OpenMM only)
  4. solvate, neutralise, add salt -> Amber prmtop/inpcrd -> GROMACS top/gro (ParmEd)
     -> Amber/GROMACS energy cross-check -> index groups (catal, loop, ...) -> staged
     position restraints (STEP4_0..4_4) -> MDPs + Slurm scripts (GROMACS, Amber)

Subcommands: prepare | finish (mcpb, after QM) | check (metal-site distances vs crystal)
"""
import argparse, json, os, re, shutil, subprocess, sys, textwrap
import numpy as np

# absolute launcher path, so generated batch scripts work without "module load metalprep"
MP_EXE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bin", "metalprep")

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "metalprep_data")
DONORS = ("N", "O", "S")
# Li/Merz ion parameter sets available per water model in AmberTools 25
ION_FRCMOD = {"hfe": "frcmod.ions234lm_hfe_{w}", "iod": "frcmod.ions234lm_iod_{w}",
              "cm": "frcmod.ions234lm_126_{w}", "1264": "frcmod.ions234lm_1264_{w}"}
WATER_LEAPRC = {"tip3p": "tip3p", "spce": "spce", "tip4pew": "tip4pew", "opc": "opc"}
WATER_BOX = {"tip3p": "TIP3PBOX", "spce": "SPCBOX", "tip4pew": "TIP4PEWBOX", "opc": "OPCBOX"}
WATER_MCPB = {"tip3p": "TIP3P", "spce": "SPCE", "tip4pew": "TIP4PEW", "opc": "OPC"}


def log(msg):
    print(f"[metalprep] {msg}", flush=True)


def die(msg):
    print(f"[metalprep] ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def run(cmd, cwd, logname=None, stdin=None, check=True):
    """Run a shell command in cwd; output to logname (appended) or inherited."""
    out = open(os.path.join(cwd, logname), "a") if logname else None
    r = subprocess.run(cmd, cwd=cwd, shell=True, input=stdin, text=True,
                       stdout=out or None, stderr=subprocess.STDOUT if out else None)
    if out:
        out.close()
    if check and r.returncode != 0:
        die(f"command failed ({r.returncode}): {cmd}" + (f"  -> see {os.path.join(cwd, logname)}" if logname else ""))
    return r.returncode


# --------------------------------------------------------------------------- PDB handling
class Atom:
    __slots__ = ("rec", "serial", "name", "alt", "resn", "chain", "resi", "icode", "xyz", "elem", "line")

    def __init__(self, line):
        self.line = line.rstrip("\n")
        self.rec = line[:6].strip()
        self.serial = int(line[6:11])
        self.name = line[12:16].strip()
        self.alt = line[16]
        self.resn = line[17:20].strip()
        self.chain = line[21]
        self.resi = int(line[22:26])
        self.icode = line[26]
        self.xyz = np.array([float(line[30:38]), float(line[38:46]), float(line[46:54])])
        el = line[76:78].strip() if len(line) >= 78 else ""
        self.elem = (el or re.sub(r"[^A-Za-z]", "", self.name)[:1]).capitalize()

    def key(self):
        return (self.chain, self.resi, self.icode)


def read_pdb(path):
    return [Atom(l) for l in open(path) if l.startswith(("ATOM  ", "HETATM"))]


def write_pdb(atoms, path, ter_after=None):
    """Write atoms; TER between chains and before/after each atom whose residue key is in ter_after."""
    with open(path, "w") as f:
        prev = None
        for a in atoms:
            if prev is not None and (a.chain != prev.chain or a.key() in (ter_after or ()) or prev.key() in (ter_after or ())) \
                    and a.key() != prev.key():
                f.write("TER\n")
            f.write(a.line[:80] + "\n")
            prev = a
        f.write("TER\nEND\n")


def set_resname(line, resn):
    return line[:17] + f"{resn:>3s}" + line[20:]


# --------------------------------------------------------------------------- site analysis
def analyse_sites(atoms, metals, cutoff):
    """Return list of sites: metal atom + ligating atoms (within cutoff, N/O/S, not H)."""
    xyz = np.array([a.xyz for a in atoms])
    sites = []
    for i, a in enumerate(atoms):
        if a.resn not in metals:
            continue
        d = np.linalg.norm(xyz - a.xyz, axis=1)
        ligs = []
        for j in np.argsort(d):
            if j == i or d[j] > cutoff:
                continue
            b = atoms[j]
            if b.elem in DONORS and b.resn not in metals:
                ligs.append({"resn": b.resn, "resi": b.resi, "chain": b.chain, "atom": b.name,
                             "elem": b.elem, "dist": round(float(d[j]), 3)})
        sites.append({"metal": a.resn, "atom": a.name, "resi": a.resi, "chain": a.chain,
                      "serial": a.serial, "xyz": [round(float(x), 3) for x in a.xyz], "ligands": ligs})
    return sites


def site_protonation(sites, overrides):
    """Decide residue names for ligating residues. Returns {(chain,resi): resname} and notes."""
    names, notes = {}, []
    for s in sites:
        for L in s["ligands"]:
            k = (L["chain"], L["resi"])
            if L["resn"] in ("HIS", "HID", "HIE", "HIP"):
                want = "HIE" if L["atom"] == "ND1" else "HID" if L["atom"] == "NE2" else None
                if want is None:
                    continue  # backbone N/O of a His
                if k in names and names[k] != want:
                    notes.append(f"His {L['resi']} binds metals through both ND1 and NE2; set it with --his")
                names[k] = want
            elif L["resn"] in ("CYS", "CYX", "CYM") and L["atom"] == "SG":
                names[k] = "CYM"
    for (ch, ri), rn in overrides.items():
        names[(ch, ri)] = rn
        notes.append(f"user override: {ch}{ri} -> {rn}")
    return names, notes


def apply_protonation(atoms, names):
    """Rename residues and drop the H that sits on a ligating atom."""
    drop = {"HIE": {"HD1"}, "HID": {"HE2"}, "CYM": {"HG"}}
    out = []
    for a in atoms:
        rn = names.get((a.chain, a.resi))
        if rn:
            if a.name in drop.get(rn, ()):
                continue
            if rn == "HID" and a.name == "HD1" and False:
                pass
            a.line = set_resname(a.line, rn)
            a.resn = rn
        out.append(a)
    return out


def site_report(sites, names, notes, path):
    lines = []
    for s in sites:
        lig = sorted(s["ligands"], key=lambda L: L["dist"])
        res = {}
        for L in lig:
            res.setdefault((L["chain"], L["resi"], L["resn"]), []).append(f"{L['atom']} {L['dist']:.2f}")
        lines.append(f"{s['metal']} {s['chain']}{s['resi']} (atom {s['serial']}): CN = {len(lig)} donor atoms, "
                     f"{len(res)} residues")
        for (ch, ri, rn), at in res.items():
            new = names.get((ch, ri), rn)
            lines.append(f"    {rn:>3s} {ch}{ri:<5d} -> {new:<4s} {', '.join(at)}")
        if len(res) < 4:
            lines.append("    note: fewer than 4 coordinating residues (open site: crystallographic water/substrate removed?)")
    lines += [f"note: {n}" for n in notes]
    open(path, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


# --------------------------------------------------------------------------- tleap helpers
def leap_head(args, extra=""):
    w = args.water
    head = f"source leaprc.protein.{args.ff}\nsource leaprc.water.{WATER_LEAPRC[w]}\n"
    return head + extra


def solvate_lines(args, unit):
    box = WATER_BOX[args.water]
    cmd = "solvateOct" if args.box_shape == "oct" else "solvateBox"
    return f"{cmd} {unit} {box} {args.box:.1f}\n"


def salt_pairs(work, build_text, unit, conc):
    """Number of NaCl pairs for conc (mol/L): N = c * N_wat / 55.5, water count from a solvation pass."""
    if conc <= 0:
        return 0
    run("tleap -f leap_saltcount.in", work, "leap_saltcount.log", check=False) if False else None
    open(os.path.join(work, "leap_saltcount.in"), "w").write(build_text + f"savepdb {unit} leap_saltcount.pdb\nquit\n")
    run("tleap -f leap_saltcount.in", work, "leap_saltcount.log")
    nwat = sum(1 for l in open(os.path.join(work, "leap_saltcount.pdb"))
               if l.startswith(("ATOM", "HETATM")) and l[17:20] == "WAT" and l[12:16].strip() == "O")
    os.remove(os.path.join(work, "leap_saltcount.pdb"))
    return int(round(conc * nwat / 55.5))


def ion_lines(unit, npairs):
    s = f"addions {unit} Na+ 0\naddions {unit} Cl- 0\n"
    if npairs > 0:
        s += f"addIonsRand {unit} Na+ {npairs} Cl- {npairs}\n"
    return s


def check_leap(work, logname):
    txt = open(os.path.join(work, logname)).read()
    m = re.findall(r"Exiting LEaP: Errors = (\d+)", txt)
    if not m or int(m[-1]) > 0:
        die(f"tleap reported errors, see {os.path.join(work, logname)}")
    if "FATAL" in txt:
        die(f"tleap FATAL, see {os.path.join(work, logname)}")


# --------------------------------------------------------------------------- models
def build_nonbonded(args, work, pdb):
    w = args.water
    model = "1264" if args.model == "nb1264" else args.ion_set
    frc = ION_FRCMOD[model].format(w=w)
    build = leap_head(args, f"loadamberparams {frc}\nmol = loadpdb {pdb}\n") + solvate_lines(args, "mol")
    npairs = salt_pairs(work, build, "mol", args.salt)
    leap = (leap_head(args, f"loadamberparams {frc}\nmol = loadpdb {pdb}\n")
            + "saveamberparm mol dry.prmtop dry.inpcrd\nsavepdb mol dry.pdb\n"
            + solvate_lines(args, "mol") + ion_lines("mol", npairs)
            + "saveamberparm mol system.prmtop system.inpcrd\nsavepdb mol system.pdb\nquit\n")
    open(os.path.join(work, "leap_build.in"), "w").write(leap)
    run("tleap -f leap_build.in", work, "leap_build.log")
    check_leap(work, "leap_build.log")
    log(f"tleap: {frc}, salt {args.salt} M -> {npairs} NaCl pairs")
    if args.model == "nb1264":
        mask = ",".join(sorted({f"@{m}" if False else f":{m}" for m in args.metal_names}))
        pol = os.path.join(os.environ.get("AMBERHOME", ""), "dat/leap/parm/lj_1264_pol.dat")
        cmds = (f"add12_6_4 {mask} watermodel {WATER_MCPB[w]} polfile {pol}\n"
                "outparm system_1264.prmtop\nquit\n")
        open(os.path.join(work, "parmed_1264.in"), "w").write(cmds)
        run("parmed -p system.prmtop -i parmed_1264.in", work, "parmed_1264.log")
        shutil.move(os.path.join(work, "system_1264.prmtop"), os.path.join(work, "system.prmtop"))
        log("12-6-4: C4 terms added with parmed add12_6_4 (system.prmtop); GROMACS export is not possible")


ZAFF_CENTRES = {
    # centre id: (composition key, residue names in order of the key, metal residue name)
    1: ("CCCC", {"C": "CY1"}, "ZN1"),
    2: ("CCCe", {"C": "CY2", "e": "HE1"}, "ZN2"),
    3: ("CCCd", {"C": "CY3", "d": "HD1"}, "ZN3"),
    4: ("CCdd", {"C": "CY4", "d": "HD2"}, "ZN4"),
    5: ("Cddd", {"C": "CY5", "d": "HD3"}, "ZN5"),
    6: ("Wdde", {"d": ["HD4", "HD5"], "e": "HE2", "W": "WT1"}, "ZN6"),
    7: ("Oddе".replace("е", "e"), {"d": ["HD6", "HD7"], "e": "HE3", "O": "OH1"}, "ZN7"),
    8: ("DDee", {"D": "AP1", "e": "HE4"}, "ZN8"),
    9: ("Ddde", {"D": "AP2", "d": ["HD8", "HD9"], "e": "HE5"}, "ZN9"),
    10: ("dddd", {"d": "HDD"}, "Z10"),
}


def zaff_templates():
    """Atom names per ZAFF residue from ZAFF.prep."""
    out, cur = {}, None
    if not all(os.path.isfile(os.path.join(DATA, f)) for f in ("ZAFF.prep", "ZAFF.frcmod")):
        die(f"ZAFF.prep/ZAFF.frcmod not found in {DATA}; run {DATA}/get_zaff.sh once")
    lines = open(os.path.join(DATA, "ZAFF.prep")).read().split("\n")
    for i, l in enumerate(lines):
        if " INT " in l:
            cur = l.split()[0]; out[cur] = set(); continue
        p = l.split()
        if cur and len(p) >= 10 and p[0].isdigit() and p[1] != "DUMM":
            out[cur].add(p[1])
        elif cur and not l.strip() and out[cur]:
            cur = None
    return out


def zaff_code(resn):
    return {"CYM": "C", "HID": "d", "HIE": "e", "ASP": "D", "WAT": "W", "HOH": "W"}.get(resn)


def build_zaff(args, work, atoms, sites):
    """Rename residues to ZAFF names, bond Zn to ligating atoms, build with tleap."""
    bonds, rename, metal_rename = [], {}, {}
    for s in sites:
        if s["metal"] != "ZN":
            die("ZAFF covers Zn only")
        res = {}
        for L in s["ligands"]:
            res.setdefault((L["chain"], L["resi"]), []).append(L)
        codes = []
        for k, Ls in res.items():
            c = zaff_code(Ls[0]["resn"])
            if c is None:
                die(f"ZAFF: residue {Ls[0]['resn']}{k[1]} at the Zn site has no ZAFF type; use --model mcpb")
            codes.append((c, k, min(Ls, key=lambda L: L["dist"])))
        key = "".join(sorted(c for c, _, _ in codes))
        match = [cid for cid, (comp, _, _) in ZAFF_CENTRES.items() if "".join(sorted(comp)) == key]
        if len(res) != 4 or not match:
            die(f"Zn {s['chain']}{s['resi']}: site {key or '-'} ({len(res)} residues) is not a ZAFF centre "
                f"(ZAFF = 4-coordinate Zn: CCCC, CCCH, CCHH, CHHH, HHH+W, DDHH, DHHH, HHHH); use --model mcpb or nb*")
        cid = match[0]
        _, rmap, zn_name = ZAFF_CENTRES[cid]
        used = {}
        for c, k, L in sorted(codes, key=lambda t: t[1][1]):  # lower residue number gets the first name
            nm = rmap[c]
            if isinstance(nm, list):
                i = used.get(c, 0); used[c] = i + 1; nm = nm[min(i, len(nm) - 1)]
            rename[k] = nm
            bonds.append((s, k, L["atom"]))
        metal_rename[(s["chain"], s["resi"])] = zn_name
        log(f"ZAFF centre {cid} for Zn {s['chain']}{s['resi']} ({key})"
            + ("  [two HID: lower residue number assigned first; check vs ZAFF note]" if cid in (6, 7, 9) else ""))
    tmpl = zaff_templates()
    kept = []
    for a in atoms:
        k = (a.chain, a.resi)
        rn = rename.get(k) or metal_rename.get(k)
        if rn:
            a.line = set_resname(a.line, rn); a.resn = rn
            names = tmpl[rn]
            if a.name not in names:
                alias = {"H": "HN"}.get(a.name)
                if alias in names:
                    a.line = a.line[:12] + f" {alias:<3s}" + a.line[16:]; a.name = alias
                elif a.elem == "H":
                    log(f"ZAFF: dropping {a.name} of {rn}{a.resi} (not in the ZAFF template)")
                    continue
                else:
                    die(f"ZAFF: atom {a.name} of {rn}{a.resi} is not in the ZAFF template {sorted(names)}")
        kept.append(a)
    atoms[:] = kept
    zn_keys = {(c, r, " ") for (c, r) in metal_rename}
    write_pdb(atoms, os.path.join(work, "zaff_input.pdb"), ter_after=zn_keys)
    # residue numbers as tleap sees them = order of residues in the file
    order, seen = {}, []
    for a in atoms:
        if (a.chain, a.resi) not in order:
            order[(a.chain, a.resi)] = len(order) + 1
    types = [l.split()[0] for l in open(os.path.join(DATA, "ZAFF.frcmod")).read().split("\nMASS\n")[1].split("\n\n")[0].splitlines() if l.strip()]
    el = {"Z": "Zn", "S": "S", "N": "N", "O": "O", "E": "O", "D": "O"}
    addtypes = "addAtomTypes {\n" + "\n".join(f'  {{ "{t}" "{el.get(t[0], t[0])}" "sp3" }}' for t in types) + "\n}\n"
    bond_txt = "".join(f"bond mol.{order[(s['chain'], s['resi'])]}.ZN mol.{order[k]}.{atom}\n" for s, k, atom in bonds)
    pre = leap_head(args, addtypes + f"loadamberprep {DATA}/ZAFF.prep\nloadamberparams {DATA}/ZAFF.frcmod\n"
                    f"mol = loadpdb zaff_input.pdb\n" + bond_txt)
    npairs = salt_pairs(work, pre + solvate_lines(args, "mol"), "mol", args.salt)
    leap = (pre + "saveamberparm mol dry.prmtop dry.inpcrd\nsavepdb mol dry.pdb\n" + solvate_lines(args, "mol")
            + ion_lines("mol", npairs) + "saveamberparm mol system.prmtop system.inpcrd\nsavepdb mol system.pdb\nquit\n")
    open(os.path.join(work, "leap_build.in"), "w").write(leap)
    run("tleap -f leap_build.in", work, "leap_build.log")
    check_leap(work, "leap_build.log")
    log(f"tleap (ZAFF): salt {args.salt} M -> {npairs} NaCl pairs")


def mcpb_step1(args, work, clean_pdb, sites):
    metals = sorted({s["metal"] for s in sites})
    ion_ids = " ".join(str(s["serial"]) for s in sites)
    info, mol2s = [], []
    for m in metals:
        q = args.metal_charge[m]
        el = m.capitalize() if len(m) > 1 else m
        info.append(f"{m} {m} {el} {q}")
        mpdb = os.path.join(work, f"{m}.pdb")
        with open(mpdb, "w") as f:
            for l in open(os.path.join(work, clean_pdb)):
                if l.startswith(("ATOM", "HETATM")) and l[17:20].strip() == m:
                    f.write(l)
        run(f"metalpdb2mol2.py -i {m}.pdb -o {m}.mol2 -c {q}", work, "mcpb.log")
        mol2s.append(f"{m}.mol2")
    inp = textwrap.dedent(f"""\
        original_pdb {clean_pdb}
        group_name MOL
        cut_off {args.cutoff}
        ion_ids {ion_ids}
        ion_info {' '.join(info)}
        ion_mol2files {' '.join(mol2s)}
        software_version g16
        force_field {args.ff}
        water_model {WATER_MCPB[args.water]}
        large_opt {1 if args.large_opt else 0}
        """)
    open(os.path.join(work, "mcpb.in"), "w").write(inp)
    env = f"MCPB_GAU_MEM={args.qm_mem} MCPB_GAU_NPROC={args.qm_cores} "
    run(env + "MCPB.py -i mcpb.in -s 1", work, "mcpb.log")
    for f in ("MOL_small_opt.com", "MOL_small_fc.com", "MOL_large_mk.com"):
        if not os.path.exists(os.path.join(work, f)):
            die(f"MCPB step 1 did not write {f}; see {work}/mcpb.log")
    # spin multiplicity: MCPB writes 1 for closed shell; open-shell metals need --mult
    if args.mult:
        for f in ("MOL_small_opt.com", "MOL_small_fc.com", "MOL_large_mk.com"):
            p = os.path.join(work, f); t = open(p).read().split("\n")
            for i, l in enumerate(t):
                if re.match(r"^\s*-?\d+\s+\d+\s*$", l):
                    t[i] = f"{l.split()[0]} {args.mult}"; break
            open(p, "w").write("\n".join(t))


def qm_scripts(args, work):
    """Gaussian jobs as a dependency chain: opt -> fc ; mk independent ; finish after both."""
    mod = f"module load {args.gaussian_module}"
    common = textwrap.dedent(f"""\
        #!/bin/bash
        #SBATCH -p main
        #SBATCH -w {args.qm_node}
        #SBATCH --ntasks=1
        #SBATCH -c {args.qm_cores}
        #SBATCH --mem={int(re.sub('[^0-9]', '', args.qm_mem)) + 8}G
        #SBATCH -t {args.qm_time}
        {mod}
        export GAUSS_SCRDIR=/tmp/$USER/g16_$SLURM_JOB_ID; mkdir -p $GAUSS_SCRDIR
        trap 'rm -rf $GAUSS_SCRDIR' EXIT
        cd {work}
        """)
    jobs = {
        "qm_opt.sbatch": "#SBATCH -J mp_opt\n" + "g16 < MOL_small_opt.com > MOL_small_opt.log\nformchk MOL_small_opt.chk MOL_small_opt.fchk\n",
        "qm_fc.sbatch": "#SBATCH -J mp_fc\n" + "g16 < MOL_small_fc.com > MOL_small_fc.log\nformchk MOL_small_opt.chk MOL_small_opt.fchk\n",
        "qm_mk.sbatch": "#SBATCH -J mp_mk\n" + "g16 < MOL_large_mk.com > MOL_large_mk.log\n",
    }
    for n, body in jobs.items():
        lines = common.split("\n")
        lines.insert(1, body.split("\n")[0])
        open(os.path.join(work, n), "w").write("\n".join(lines) + "\n".join(body.split("\n")[1:]))
    fin = textwrap.dedent(f"""\
        #!/bin/bash
        #SBATCH -J mp_finish
        #SBATCH -p main
        #SBATCH -w {args.qm_node}
        #SBATCH --ntasks=1
        #SBATCH -c 4
        #SBATCH -t 02:00:00
        cd {work}
        {MP_EXE} finish {work}
        """)
    open(os.path.join(work, "finish.sbatch"), "w").write(fin)
    sub = textwrap.dedent("""\
        #!/bin/bash
        # Submits the Gaussian chain: opt -> fc (needs the optimised checkpoint), mk in parallel, then finish.
        set -e
        cd "$(dirname "$0")"
        o=$(sbatch --parsable qm_opt.sbatch)
        f=$(sbatch --parsable --dependency=afterok:$o qm_fc.sbatch)
        m=$(sbatch --parsable qm_mk.sbatch)
        z=$(sbatch --parsable --dependency=afterok:$f:$m finish.sbatch)
        echo "submitted: opt $o, fc $f, mk $m, finish $z"
        """)
    p = os.path.join(work, "submit_qm.sh")
    open(p, "w").write(sub)
    os.chmod(p, 0o755)


def mcpb_finish(args, work):
    if not all(os.path.exists(os.path.join(work, f)) for f in ("MOL_small_opt.fchk", "MOL_small_fc.log", "MOL_large_mk.log")):
        die("Gaussian outputs missing (MOL_small_opt.fchk, MOL_small_fc.log, MOL_large_mk.log)")
    for f in ("MOL_small_fc.log", "MOL_large_mk.log", "MOL_small_opt.log"):
        if "Normal termination" not in open(os.path.join(work, f)).read()[-3000:]:
            die(f"{f} did not terminate normally")
    for s in ("2", "3", "4"):
        run(f"MCPB.py -i mcpb.in -s {s}", work, "mcpb.log")
    t = open(os.path.join(work, "MOL_tleap.in")).read()
    # keep MCPB's parameter/residue/bond section; replace solvation, ions and outputs by ours
    keep = []
    for l in t.splitlines():
        if re.match(r"^\s*(solvate|addions|addIons|savepdb|saveamberparm|quit)", l, re.I):
            continue
        keep.append(l)
    pre = "\n".join(keep) + "\n"
    unit = re.search(r"^\s*(\w+)\s*=\s*loadpdb", t, re.M | re.I).group(1)
    npairs = salt_pairs(work, pre + solvate_lines(args, unit), unit, args.salt)
    leap = (pre + f"saveamberparm {unit} dry.prmtop dry.inpcrd\nsavepdb {unit} dry.pdb\n" + solvate_lines(args, unit)
            + ion_lines(unit, npairs) + f"saveamberparm {unit} system.prmtop system.inpcrd\nsavepdb {unit} system.pdb\nquit\n")
    open(os.path.join(work, "leap_build.in"), "w").write(leap)
    run("tleap -f leap_build.in", work, "leap_build.log")
    check_leap(work, "leap_build.log")
    log(f"tleap (MCPB): salt {args.salt} M -> {npairs} NaCl pairs")


# --------------------------------------------------------------------------- post-processing
WATERS = ("WAT", "HOH", "SOL")
IONS = {"Na+": "NA", "NA+": "NA", "Na": "NA", "Cl-": "CL", "CL-": "CL", "Cl": "CL"}


def split_cmap_types(s):
    """ParmEd writes GROMACS cmaptypes keyed on atom types; ff19SB has residue-specific CMAPs on the same
    types (C N XC C N), so they collapse into one. Give each CMAP class its own copy of the central
    atom type (identical mass/LJ; bonded terms are written inline) so every class keeps its own grid."""
    import copy
    cls, new = {}, {}
    for c in s.cmaps:
        k = cls.setdefault(id(c.type), len(cls))
        a = c.atom3
        old = a.atom_type
        key = (old.name, k)
        if key not in new:
            nt = copy.copy(old)
            nt.name = f"{old.name}{k}"
            nt._bond_type = nt.name
            new[key] = nt
        a.atom_type = new[key]
        a.type = new[key].name
    return len(cls)


def gromacs_save(s, top, gro):
    for r in s.residues:
        rn = r.name.strip()
        if rn in ("WAT", "HOH"):
            r.name = "SOL"
        elif rn in IONS:
            r.name = IONS[rn]
    n = split_cmap_types(s) if s.cmaps else 0
    s.save(top, format="gromacs", overwrite=True)
    s.save(gro, overwrite=True)
    return n


def to_gromacs(work):
    import parmed
    s = parmed.load_file(os.path.join(work, "system.prmtop"), os.path.join(work, "system.inpcrd"))
    n = gromacs_save(s, os.path.join(work, "system.top"), os.path.join(work, "system.gro"))
    log("GROMACS: system.top / system.gro written (ParmEd" + (f"; {n} residue-specific CMAP classes kept)" if n else ")"))


def energy_check(work):
    """Solute (dry) single-point energy by component, Amber prmtop vs the GROMACS conversion (OpenMM, no cutoff).
    Water/ions are standard; the solute (metal site, CMAP, renamed residues) is where conversion can go wrong."""
    import parmed, openmm as mm, openmm.app as app, openmm.unit as u
    def comps(x):
        sysm = x.createSystem(nonbondedMethod=app.NoCutoff, constraints=None)
        for i, f in enumerate(sysm.getForces()):
            f.setForceGroup(i)
        ctx = mm.Context(sysm, mm.VerletIntegrator(1e-3), mm.Platform.getPlatformByName("CPU"))
        ctx.setPositions(x.positions)
        return {type(f).__name__: ctx.getState(getEnergy=True, groups={i}).getPotentialEnergy()
                .value_in_unit(u.kilojoule_per_mole) for i, f in enumerate(sysm.getForces())}
    amb = parmed.load_file(os.path.join(work, "dry.prmtop"), os.path.join(work, "dry.inpcrd"))
    tmp = parmed.load_file(os.path.join(work, "dry.prmtop"), os.path.join(work, "dry.inpcrd"))
    gromacs_save(tmp, os.path.join(work, "dry_check.top"), os.path.join(work, "dry_check.gro"))
    gmx = parmed.load_file(os.path.join(work, "dry_check.top"), xyz=os.path.join(work, "dry_check.gro"))
    gmx.coordinates = amb.coordinates  # .gro keeps 0.001 nm only; compare at identical coordinates
    ca, cg = comps(amb), comps(gmx)
    lines, worst = [], 0.0
    for k in sorted(set(ca) | set(cg)):
        a, g = ca.get(k, 0.0), cg.get(k, 0.0)
        d = abs(a - g)
        worst = max(worst, d)
        lines.append(f"  {k:<24s} Amber {a:14.3f}  GROMACS {g:14.3f}  diff {d:8.3f} kJ/mol")
    msg = "Solute energy, Amber prmtop vs GROMACS topology (OpenMM, no cutoff):\n" + "\n".join(lines)
    print(msg)
    open(os.path.join(work, "energy_check.txt"), "w").write(msg + "\n")
    for f in ("dry_check.top", "dry_check.gro"):
        os.remove(os.path.join(work, f))
    if worst > 0.5:
        die(f"Amber and GROMACS solute energies differ by {worst:.2f} kJ/mol in one component -- conversion problem")


def parse_top_molecules(top):
    """Return [(molname, natoms)] per moleculetype definition and [(molname, count)] from [ molecules ]."""
    defs, mols, sec, cur = {}, [], None, None
    for l in open(top):
        s = l.split(";")[0].strip()
        if not s:
            continue
        m = re.match(r"\[\s*(\w+)\s*\]", s)
        if m:
            sec = m.group(1); continue
        if sec == "moleculetype":
            cur = s.split()[0]; defs[cur] = 0; sec = "_mt_named"
        elif sec == "atoms":
            defs[cur] += 1
        elif sec == "molecules":
            n, c = s.split()[:2]; mols.append((n, int(c)))
    return defs, mols


def metal_atoms(struct, sites):
    """Indices of single-atom metal residues, matched by element (MCPB/ZAFF rename them, e.g. ZN1, Z10)."""
    import parmed
    zs = {parmed.periodic_table.AtomicNum[re.sub(r"[^A-Za-z]", "", s["atom"]).capitalize()] for s in sites}
    return [a.idx for a in struct.atoms if a.element in zs and len(a.residue.atoms) == 1]


def index_and_restraints(args, work, sites):
    """Write index.ndx (named groups) and staged position restraints like the 4LXC protocol."""
    import parmed
    s = parmed.load_file(os.path.join(work, "system.prmtop"), os.path.join(work, "system.inpcrd"))
    metal_res = set()
    lig_res = set()
    zres = [s.atoms[i].residue for i in metal_atoms(s, sites)]
    # map crystal metal sites to residues by position
    for site in sites:
        x = np.array(site["xyz"])
        best = min(zres, key=lambda r: np.linalg.norm(np.array([r.atoms[0].xx, r.atoms[0].xy, r.atoms[0].xz]) - x))
        metal_res.add(best.idx)
    xyz = np.array([[a.xx, a.xy, a.xz] for a in s.atoms])
    for ri in metal_res:
        m = s.residues[ri].atoms[0]
        d = np.linalg.norm(xyz - np.array([m.xx, m.xy, m.xz]), axis=1)
        for j in np.where(d < args.cutoff)[0]:
            a = s.atoms[j]
            if a.element_name in DONORS and a.residue.idx not in metal_res and a.residue.name.strip() not in WATERS:
                lig_res.add(a.residue.idx)
    water_like = set(WATERS) | set(IONS.values()) | set(IONS) | {"SOL"}
    prot = [a.idx for a in s.atoms if a.residue.name.strip() not in water_like and a.residue.idx not in metal_res]
    heavy = lambda idx: [i for i in idx if s.atoms[i].element != 1]
    metal = [a.idx for r in metal_res for a in s.residues[r].atoms]
    catal = [a.idx for r in sorted(lig_res | metal_res) for a in s.residues[r].atoms]
    loop = []
    if args.loop:
        lo, hi = args.loop_new
        loop = [a.idx for a in s.atoms if a.residue.idx not in metal_res and lo <= a.residue.number <= hi
                and a.residue.name.strip() not in water_like]
    sol = [a.idx for a in s.atoms if a.residue.name.strip() in WATERS]
    ions = [a.idx for a in s.atoms if a.residue.name.strip() in IONS or a.residue.name.strip() in ("NA", "CL")]
    groups = {
        "System": list(range(len(s.atoms))),
        "Protein": prot, "Protein-H": heavy(prot),
        "Metal": metal, f"Protein_{'_'.join(sorted(args.metal_names))}": sorted(prot + metal),
        "catal": catal, "catal-H": heavy(catal),
        "loop": loop, "loop-H": heavy(loop),
        "Protein_H_nocatal_noloop": [i for i in heavy(prot) if i not in set(catal) | set(loop)],
        "Water_and_ions": sorted(sol + ions), "SOL": sol, "Ions": ions,
    }
    with open(os.path.join(work, "index.ndx"), "w") as f:
        for n, idx in groups.items():
            if not idx and n.startswith("loop"):
                continue
            f.write(f"[ {n} ]\n")
            for i in range(0, len(idx), 15):
                f.write(" ".join(f"{k + 1:>6d}" for k in idx[i:i + 15]) + "\n")
    # per-atom restraint tag (heavy atoms only, like the original scheme)
    tag = {}
    for i in groups["Protein_H_nocatal_noloop"]:
        tag[i] = "fc_proH"
    for i in heavy(catal):
        tag[i] = "fc_catalH"
    for i in heavy(loop):
        if i not in set(catal):
            tag[i] = "fc_loopH"
    header = textwrap.dedent("""\
        ; staged restraints (metalprep): select with -DPOSRES -DSTEP4_n ; -DCATAL / -DLOOP keep those groups at 4000
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
        #undef fc_catalH
        #define fc_catalH 4000
        #endif
        #ifdef LOOP
        #undef fc_loopH
        #define fc_loopH 4000
        #endif
        [ position_restraints ]
        ;    i funct        fcx        fcy        fcz
        """)
    top = os.path.join(work, "system.top")
    defs, mols = parse_top_molecules(top)
    start, done, includes = 0, set(), {}
    for name, count in mols:
        n = defs[name]
        if name not in done:
            body = [(i - start + 1, tag[i]) for i in range(start, start + n) if i in tag]
            if body:
                fn = f"posre_{name}.itp"
                with open(os.path.join(work, fn), "w") as f:
                    f.write(header)
                    for li, t in body:
                        f.write(f"{li:10d} {1:5d} {t:>10s} {t:>10s} {t:>10s}\n")
                includes[name] = fn
            done.add(name)
        start += n * count
    # insert '#ifdef POSRES #include' at the end of each restrained moleculetype block
    out, cur, lines = [], None, open(top).read().split("\n")
    def flush():
        if cur in includes:
            out.extend(["#ifdef POSRES", f'#include "{includes[cur]}"', "#endif", ""])
    i = 0
    while i < len(lines):
        l = lines[i]
        m = re.match(r"\s*\[\s*(\w+)\s*\]", l)
        if m and m.group(1) in ("moleculetype", "system"):
            flush(); cur = None
            if m.group(1) == "moleculetype":
                out.append(l); i += 1
                while i < len(lines) and (not lines[i].strip() or lines[i].strip().startswith(";")):
                    out.append(lines[i]); i += 1
                cur = lines[i].split()[0]
        out.append(lines[i]); i += 1
    open(top, "w").write("\n".join(out))
    log(f"index.ndx groups: catal = {len(lig_res)} residues + {len(metal_res)} metal(s); restraints in "
        f"{', '.join(includes.values())}")
    return groups


def write_md_files(args, work):
    """MDPs (from the 4LXC protocol) + Slurm scripts for GROMACS (and Amber pmemd.cuda)."""
    mdpdir = os.path.join(work, "MDP")
    os.makedirs(mdpdir, exist_ok=True)
    for f in os.listdir(os.path.join(DATA, "mdp")):
        t = open(os.path.join(DATA, "mdp", f)).read()
        t = re.sub(r"(?m)^(\s*ref_t\s*=\s*)[0-9.]+", rf"\g<1>{args.temperature:g}", t)
        open(os.path.join(mdpdir, f), "w").write(t)
    gpu = args.md_node == "cws01"
    gmxmod = "gromacs/2026.1-tmpi-cuda" if gpu else "gromacs/2026.1-tmpi"
    res = "#SBATCH --gres=gpu:1\n#SBATCH -c 8" if gpu else "#SBATCH -c 32"
    mdrun = "gmx mdrun -ntmpi 1 -ntomp $SLURM_CPUS_PER_TASK -nb gpu -pme gpu -bonded gpu -update gpu" if gpu \
        else "gmx mdrun -ntmpi 1 -ntomp $SLURM_CPUS_PER_TASK"
    mdrun_em = "gmx mdrun -ntmpi 1 -ntomp $SLURM_CPUS_PER_TASK"
    gp = "gmx grompp -n index.ndx -p system.top -maxwarn 1"
    mini = textwrap.dedent(f"""\
        #!/bin/bash
        #SBATCH -J mp_min
        #SBATCH -p main
        #SBATCH -w {args.md_node}
        #SBATCH --ntasks=1
        @RES@
        #SBATCH -t 04:00:00
        #SBATCH -o min_%j.out
        # Staged minimisation (4LXC protocol): steep+cg, then steep+cg at STEP4_0..4_4 (restraints 4000 -> 0)
        set -euo pipefail
        module load {gmxmod}
        export GMX_MAXBACKUP=-1 CUDA_DEVICE_ORDER=PCI_BUS_ID
        cd "$SLURM_SUBMIT_DIR"
        {gp} -f MDP/em_steep.mdp -c system.gro -r system.gro -o em_steep.tpr
        {mdrun_em} -deffnm em_steep
        {gp} -f MDP/em_cg.mdp -c em_steep.gro -r em_steep.gro -o em_cg.tpr
        {mdrun_em} -deffnm em_cg
        prev=em_cg
        for i in 0 1 2 3 4; do
          {gp} -f MDP/em_steep4$i.mdp -c $prev.gro -r $prev.gro -o em_steep4$i.tpr
          {mdrun_em} -deffnm em_steep4$i
          {gp} -f MDP/em_cg4$i.mdp -c em_steep4$i.gro -r em_steep4$i.gro -o em_cg4$i.tpr
          {mdrun_em} -deffnm em_cg4$i
          prev=em_cg4$i
        done
        {MP_EXE} check . em_cg44.gro || true
        """)
    md = textwrap.dedent(f"""\
        #!/bin/bash
        #SBATCH -J mp_md
        #SBATCH -p main
        #SBATCH -w {args.md_node}
        #SBATCH --ntasks=1
        @RES@
        #SBATCH -t 3-00:00:00
        #SBATCH -o md_%A_%a.out
        #SBATCH --array=0-{args.replicas - 1}%{1 if gpu else 2}
        # One replica per array task: NVT release STEP4_0..4_4 (new velocities each replica), NPT, production
        set -euo pipefail
        module load {gmxmod}
        export GMX_MAXBACKUP=-1 CUDA_DEVICE_ORDER=PCI_BUS_ID
        cd "$SLURM_SUBMIT_DIR"
        r=$SLURM_ARRAY_TASK_ID
        {gp} -f MDP/nvt50.mdp -c em_cg44.gro -r em_cg44.gro -o nvt50_$r.tpr
        {mdrun} -deffnm nvt50_$r
        for i in 1 2 3 4; do j=$((i-1))
          {gp} -f MDP/nvt5$i.mdp -c nvt5${{j}}_$r.gro -r nvt5${{j}}_$r.gro -t nvt5${{j}}_$r.cpt -o nvt5${{i}}_$r.tpr
          {mdrun} -deffnm nvt5${{i}}_$r
        done
        {gp} -f MDP/npt5.mdp -c nvt54_$r.gro -r nvt54_$r.gro -t nvt54_$r.cpt -o npt5_$r.tpr
        {mdrun} -deffnm npt5_$r
        {gp} -f MDP/md6.mdp -c npt5_$r.gro -t npt5_$r.cpt -o md6_$r.tpr
        {mdrun} -deffnm md6_$r -cpi md6_$r.cpt
        {MP_EXE} check . npt5_$r.gro || true
        """)
    submit = textwrap.dedent("""\
        #!/bin/bash
        # GROMACS: staged minimisation, then the replica array (starts when minimisation finished OK)
        set -e
        cd "$(dirname "$0")"
        m=$(sbatch --parsable run_gmx_min.sbatch)
        a=$(sbatch --parsable --dependency=afterok:$m run_gmx_md.sbatch)
        echo "submitted: minimisation $m, MD array $a"
        """)
    mini, md = mini.replace("@RES@", res), md.replace("@RES@", res)
    for n, t in (("run_gmx_min.sbatch", mini), ("run_gmx_md.sbatch", md), ("submit_gromacs.sh", submit)):
        p = os.path.join(work, n); open(p, "w").write(t); os.chmod(p, 0o755)
    # analysis (named groups, no hard-coded group numbers)
    ana = textwrap.dedent(f"""\
        #!/bin/bash
        # Usage: bash md_analysis.sh [replica]   (run inside a job or interactive node; gmx module loaded)
        set -euo pipefail
        r=${{1:-0}}; P=Protein_{'_'.join(sorted(args.metal_names))}
        echo $P | gmx trjconv -f md6_$r.xtc -s em_steep40.tpr -n index.ndx -pbc nojump -o md6_${{r}}_nojump.xtc
        printf "Protein-H\\n$P\\n" | gmx trjconv -f md6_${{r}}_nojump.xtc -s em_steep40.tpr -n index.ndx -fit rot+trans -o md6_${{r}}_fit.xtc
        rm md6_${{r}}_nojump.xtc
        echo $P | gmx gyrate -f md6_${{r}}_fit.xtc -s md6_$r.tpr -n index.ndx -o gyr_${{r}}.xvg
        echo $P | gmx rmsf -f md6_${{r}}_fit.xtc -s md6_$r.tpr -n index.ndx -res -o rmsf_res_${{r}}.xvg -ox average_${{r}}.pdb
        printf "Protein-H\\nProtein-H\\n" | gmx rms -f md6_${{r}}_fit.xtc -s em_steep40.tpr -n index.ndx -o rmsd_heavy_vs_xray_${{r}}.xvg
        printf "catal-H\\ncatal-H\\n" | gmx rms -f md6_${{r}}_fit.xtc -s em_steep40.tpr -n index.ndx -o rmsd_site_vs_xray_${{r}}.xvg
        {MP_EXE} check . md6_${{r}}_fit.xtc --tpr md6_$r.tpr || true
        """)
    p = os.path.join(work, "md_analysis.sh"); open(p, "w").write(ana); os.chmod(p, 0o755)
    # Amber pmemd.cuda route (required for 12-6-4; optional otherwise)
    mask = "!:WAT,Na+,Cl- & !@H="
    amber_in = {
        "amb_min1.in": f"min, restrained ({mask}) 10 kcal/mol/A^2\n &cntrl\n  imin=1, maxcyc=5000, ncyc=2500, ntb=1, cut=10.0,\n  ntr=1, restraint_wt=10.0, restraintmask='{mask}',\n /\n",
        "amb_min2.in": "min, unrestrained\n &cntrl\n  imin=1, maxcyc=5000, ncyc=2500, ntb=1, cut=10.0,\n /\n",
        "amb_heat.in": f"NVT heating 0->{args.temperature:g} K, restrained 5\n &cntrl\n  imin=0, irest=0, ntx=1, nstlim=50000, dt=0.002, ntc=2, ntf=2, ntb=1, cut=10.0,\n  ntt=3, gamma_ln=2.0, tempi=0.0, temp0={args.temperature:g}, ig=-1,\n  ntr=1, restraint_wt=5.0, restraintmask='{mask}',\n  ntpr=1000, ntwx=1000, nmropt=1,\n /\n &wt type='TEMP0', istep1=0, istep2=40000, value1=0.0, value2={args.temperature:g} /\n &wt type='END' /\n",
        "amb_npt.in": f"NPT, restraints released 1 -> 0 via stages run by the script\n &cntrl\n  imin=0, irest=1, ntx=5, nstlim=250000, dt=0.002, ntc=2, ntf=2, ntb=2, ntp=1, barostat=2, cut=10.0,\n  ntt=3, gamma_ln=2.0, temp0={args.temperature:g}, ig=-1, ntpr=5000, ntwx=5000,\n  ntr=1, restraint_wt=RESTWT, restraintmask='{mask}',\n /\n",
        "amb_prod.in": f"production NPT\n &cntrl\n  imin=0, irest=1, ntx=5, nstlim=10000000, dt=0.002, ntc=2, ntf=2, ntb=2, ntp=1, barostat=2, cut=10.0,\n  ntt=3, gamma_ln=2.0, temp0={args.temperature:g}, ig=-1, ntpr=5000, ntwx=5000, ntwr=500000,\n /\n",
    }
    os.makedirs(os.path.join(work, "AMBER"), exist_ok=True)
    for n, t in amber_in.items():
        open(os.path.join(work, "AMBER", n), "w").write(t)
    amb = textwrap.dedent(f"""\
        #!/bin/bash
        #SBATCH -J mp_amber
        #SBATCH -p main
        #SBATCH -w cws01
        #SBATCH --gres=gpu:1
        #SBATCH --ntasks=1
        #SBATCH -c 4
        #SBATCH -t 3-00:00:00
        #SBATCH -o amber_%A_%a.out
        #SBATCH --array=0-{args.replicas - 1}%1
        # Amber pmemd.cuda route (the only GPU route for 12-6-4); one replica per array task
        set -euo pipefail
        module load amber/26
        export CUDA_DEVICE_ORDER=PCI_BUS_ID
        cd "$SLURM_SUBMIT_DIR"; r=$SLURM_ARRAY_TASK_ID
        P=system.prmtop
        if [ ! -s min2.rst7 ]; then
          pmemd.cuda -O -i AMBER/amb_min1.in -p $P -c system.inpcrd -ref system.inpcrd -o min1.out -r min1.rst7
          pmemd.cuda -O -i AMBER/amb_min2.in -p $P -c min1.rst7 -o min2.out -r min2.rst7
        fi
        pmemd.cuda -O -i AMBER/amb_heat.in -p $P -c min2.rst7 -ref min2.rst7 -o heat_$r.out -r heat_$r.rst7 -x heat_$r.nc
        prev=heat_$r
        for w in 2.0 1.0 0.5 0.1 0.0; do
          sed "s/RESTWT/$w/" AMBER/amb_npt.in > npt_${{w}}_$r.in
          pmemd.cuda -O -i npt_${{w}}_$r.in -p $P -c $prev.rst7 -ref $prev.rst7 -o npt_${{w}}_$r.out -r npt_${{w}}_$r.rst7 -x npt_${{w}}_$r.nc
          prev=npt_${{w}}_$r
        done
        pmemd.cuda -O -i AMBER/amb_prod.in -p $P -c $prev.rst7 -o prod_$r.out -r prod_$r.rst7 -x prod_$r.nc
        """)
    p = os.path.join(work, "run_amber.sbatch"); open(p, "w").write(amb); os.chmod(p, 0o755)


def finalize(args, work, sites):
    if args.model != "nb1264":
        to_gromacs(work)
        energy_check(work)
        index_and_restraints(args, work, sites)
    write_md_files(args, work)
    if args.model == "nb1264":
        for f in ("run_gmx_min.sbatch", "run_gmx_md.sbatch", "submit_gromacs.sh", "md_analysis.sh"):
            os.remove(os.path.join(work, f))
        shutil.rmtree(os.path.join(work, "MDP"))
    summary = {
        "model": args.model, "ff": args.ff, "water": args.water, "salt_M": args.salt,
        "metal_charges": args.metal_charge, "sites": sites,
        "amber": ["system.prmtop", "system.inpcrd"],
        "gromacs": [] if args.model == "nb1264" else ["system.top", "system.gro", "index.ndx", "posre_*.itp", "MDP/"],
    }
    json.dump(summary, open(os.path.join(work, "metalprep.json"), "w"), indent=1, default=str)
    nxt = ("sbatch run_amber.sbatch   (12-6-4 runs with Amber/OpenMM only)" if args.model == "nb1264"
           else "bash submit_gromacs.sh   (or: sbatch run_amber.sbatch)")
    log(f"done. Next: cd {work} && {nxt}")


# --------------------------------------------------------------------------- prepare
def prepare(args):
    work = os.path.abspath(args.out)
    if os.path.exists(work) and os.listdir(work):
        if not args.force:
            die(f"{work} exists and is not empty (use --force to rebuild it from scratch)")
        if not os.path.exists(os.path.join(work, "metalprep_args.json")):
            die(f"--force refuses to clear {work}: it was not made by metalprep")
        shutil.rmtree(work)
    os.makedirs(work, exist_ok=True)
    if args.model == "mcpb" and not os.environ.get("SLURM_JOB_ID") and not args.no_submit:
        pass  # step 1 is light; QM runs in Slurm jobs
    # 1. input
    src = args.pdb
    if os.path.isfile(src):
        shutil.copy(src, os.path.join(work, "input.pdb"))
    elif re.fullmatch(r"[0-9A-Za-z]{4}", src):
        run(f"wget -q -O input.pdb https://files.rcsb.org/download/{src.upper()}.pdb", work)
    else:
        die(f"--pdb: not a file and not a 4-character PDB ID: {src}")
    atoms = read_pdb(os.path.join(work, "input.pdb"))
    atoms = [a for a in atoms if a.alt in (" ", "A")]
    chains = set(args.chain) if args.chain else {a.chain for a in atoms if a.rec == "ATOM"}
    keep = [a for a in atoms if a.chain in chains and (a.rec == "ATOM" or a.resn in args.metal_names)]
    if not any(a.resn in args.metal_names for a in keep):
        die(f"no metal {sorted(args.metal_names)} found in chain(s) {sorted(chains)}")
    # metal-bound crystal waters
    mx = np.array([a.xyz for a in keep if a.resn in args.metal_names])
    wat = [a for a in atoms if a.chain in chains and a.resn == "HOH" and a.elem == "O"
           and np.min(np.linalg.norm(mx - a.xyz, axis=1)) < args.cutoff]
    other = sorted({(a.resn, a.chain, a.resi) for a in atoms if a.rec == "HETATM" and a.resn not in args.metal_names
                    and a.resn != "HOH" and a.chain in chains
                    and np.min(np.linalg.norm(mx - a.xyz, axis=1)) < args.cutoff})
    for rn, ch, ri in other:
        log(f"WARNING: {rn} {ch}{ri} binds the metal in the crystal but is removed (not parametrised); "
            f"the site will be open there")
    if args.keep_site_waters:
        keep += wat
        if wat:
            log(f"keeping {len(wat)} metal-bound crystal water(s)")
    keep.sort(key=lambda a: a.serial)
    # metals after protein so pdb4amber/tleap see protein first
    keep = [a for a in keep if a.resn not in args.metal_names and a.resn != "HOH"] + \
           [a for a in keep if a.resn in args.metal_names] + [a for a in keep if a.resn == "HOH"]
    write_pdb(keep, os.path.join(work, "selected.pdb"))
    # 2. pdb4amber (+reduce)
    run("pdb4amber -i selected.pdb -o clean_reduce.pdb --reduce", work, "pdb4amber.log")
    # residue renumbering map (original chain/resi -> new resi)
    renum = {}
    for l in open(os.path.join(work, "clean_reduce_renum.txt")):
        p = l.split()
        if len(p) >= 5:
            renum[(p[1], int(p[2]))] = int(p[4])
    if args.loop:
        lo, hi = args.loop
        if args.loop_is_new:
            args.loop_new = (lo, hi)
        else:
            ch = sorted(chains)[0] if len(chains) == 1 else args.loop_chain
            try:
                args.loop_new = (renum[(ch, lo)], renum[(ch, hi)])
            except KeyError:
                first = min(r for (c, r) in renum if c == ch)
                die(f"--loop {lo}-{hi}: residues not in chain {ch} (its numbering starts at {first}); "
                    f"use PDB numbering, or new:{lo}-{hi} for the renumbered (1-based) system")
        log(f"loop group: residues {args.loop_new[0]}-{args.loop_new[1]} of the renumbered system")
    atoms = read_pdb(os.path.join(work, "clean_reduce.pdb"))
    # pdb4amber drops chain IDs; use one chain label for the renumbered structure
    sites = analyse_sites(atoms, args.metal_names, args.cutoff)
    over = {}
    for o in args.his or []:
        m = re.fullmatch(r"([A-Za-z]?)(\d+):(HID|HIE|HIP)", o)
        if not m:
            die(f"--his {o}: use [chain]resnum:HID|HIE|HIP (original numbering)")
        ch = m.group(1) or sorted(chains)[0]
        new = renum.get((ch, int(m.group(2))))
        if new is None:
            die(f"--his {o}: residue not found")
        over[(atoms[0].chain, new)] = m.group(3)
    names, notes = site_protonation(sites, over)
    atoms = apply_protonation(atoms, names)
    # waters from the crystal: tleap name WAT
    for a in atoms:
        if a.resn == "HOH":
            a.line = set_resname(a.line, "WAT"); a.resn = "WAT"
    metal_keys = {a.key() for a in atoms if a.resn in args.metal_names}
    write_pdb(atoms, os.path.join(work, "clean.pdb"), ter_after=metal_keys)
    sites = analyse_sites(read_pdb(os.path.join(work, "clean.pdb")), args.metal_names, args.cutoff)
    log("metal sites (renumbered by pdb4amber; original numbering in clean_reduce_renum.txt):")
    site_report(sites, names, notes, os.path.join(work, "site_report.txt"))
    json.dump({"sites": sites, "renum": {f"{k[0]}{k[1]}": v for k, v in renum.items()}},
              open(os.path.join(work, "sites_crystal.json"), "w"), indent=1)
    json.dump(vars(args), open(os.path.join(work, "metalprep_args.json"), "w"), indent=1, default=str)
    if args.site_only:
        log("--site-only: stopping after the site report"); return
    # 3. parameters
    if args.model in ("nb126", "nb1264"):
        build_nonbonded(args, work, "clean.pdb")
    elif args.model == "zaff":
        build_zaff(args, work, read_pdb(os.path.join(work, "clean.pdb")), sites)
    elif args.model == "mcpb":
        mcpb_step1(args, work, "clean.pdb", sites)
        qm_scripts(args, work)
        if args.no_submit:
            log(f"MCPB step 1 done. Submit the Gaussian chain with: {work}/submit_qm.sh")
        else:
            run("./submit_qm.sh", work)
            log("Gaussian chain submitted; 'metalprep finish' runs automatically when it is done")
        return
    finalize(args, work, sites)


def finish(args):
    work = os.path.abspath(args.dir)
    saved = json.load(open(os.path.join(work, "metalprep_args.json")))
    ns = argparse.Namespace(**saved)
    ns.loop_new = tuple(saved["loop_new"]) if saved.get("loop_new") else None
    sites = json.load(open(os.path.join(work, "sites_crystal.json")))["sites"]
    mcpb_finish(ns, work)
    finalize(ns, work, sites)


def check(args):
    """Metal-donor distances in a structure (gro/pdb/rst7) or trajectory (xtc/nc/dcd) vs the crystal site."""
    import parmed
    work = os.path.abspath(args.dir)
    crys = json.load(open(os.path.join(work, "sites_crystal.json")))["sites"]
    ref = parmed.load_file(os.path.join(work, "system.prmtop"), os.path.join(work, "system.inpcrd"))
    x0 = np.array(ref.coordinates)
    path = args.coords if os.path.isabs(args.coords) else os.path.join(work, args.coords)
    traj = path.endswith((".xtc", ".trr"))
    frames = None if traj else np.asarray(parmed.load_file(path).coordinates).reshape(-1, len(ref.atoms), 3)
    midx = metal_atoms(ref, crys)
    pairs, labels = [], []
    for site in crys:
        mi = min(midx, key=lambda i: np.linalg.norm(x0[i] - np.array(site["xyz"])))
        for L in site["ligands"]:
            cand = [a.idx for a in ref.atoms if a.name == L["atom"]]
            li = min(cand, key=lambda i: np.linalg.norm(x0[i] - x0[mi]))
            pairs.append((mi, li)); labels.append((site, L))
    dists = []
    if traj:  # gmx distance (module gromacs loaded); needs a tpr for PBC
        tpr = args.tpr or "em_steep40.tpr"
        with open(os.path.join(work, "check_pairs.ndx"), "w") as f:
            for k, (mi, li) in enumerate(pairs):
                f.write(f"[ pair_{k} ]\n{mi + 1} {li + 1}\n")
        sel = " ".join(f"'group \"pair_{k}\"'" for k in range(len(pairs)))
        run(f"gmx distance -f {path} -s {tpr} -n check_pairs.ndx -select {sel} -oall check_dist.xvg", work, "check.log")
        data = np.loadtxt([l for l in open(os.path.join(work, "check_dist.xvg")) if not l.startswith(("#", "@"))], ndmin=2)
        dists = [data[:, k + 1] * 10.0 for k in range(len(pairs))]
    else:
        dists = [np.linalg.norm(frames[:, li] - frames[:, mi], axis=1) for mi, li in pairs]
    out = []
    for (site, L), d in zip(labels, dists):
        if True:
            flag = "  <-- moved > 0.5 A" if abs(d.mean() - L["dist"]) > 0.5 else ""
            out.append(f"{site['metal']}{site['resi']}-{L['resn']}{L['resi']}:{L['atom']:<4s} crystal {L['dist']:.2f}  "
                       f"now {d.mean():.2f} +/- {d.std():.2f} (max {d.max():.2f}) A{flag}")
    print("\n".join(out))


def main():
    p = argparse.ArgumentParser(prog="metalprep", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("prepare", help="build a solvated metalloprotein system")
    a.add_argument("--pdb", required=True, help="PDB ID (downloaded) or local PDB file")
    a.add_argument("--chain", action="append", help="chain(s) to keep (repeatable); default all protein chains")
    a.add_argument("--model", required=True, choices=["mcpb", "zaff", "nb126", "nb1264"])
    a.add_argument("--metal", action="append", default=None,
                   help="metal residue name and charge, e.g. ZN:2 (repeatable; default ZN:2)")
    a.add_argument("--ion-set", default="cm", choices=["cm", "hfe", "iod"],
                   help="Li/Merz 12-6 set for nb126 (cm = compromise, tleap default)")
    a.add_argument("--ff", default="ff19SB", help="protein force field: ff19SB | ff14SB")
    a.add_argument("--water", default="tip3p", choices=list(WATER_LEAPRC))
    a.add_argument("--box", type=float, default=10.0, help="solvent padding in A (default 10)")
    a.add_argument("--box-shape", default="box", choices=["box", "oct"])
    a.add_argument("--salt", type=float, default=0.15, help="NaCl mol/L on top of neutralising ions")
    a.add_argument("--cutoff", type=float, default=2.8, help="metal-donor bond cutoff in A (MCPB default 2.8)")
    a.add_argument("--his", action="append", help="override, e.g. 279:HID or B279:HIE (original numbering)")
    a.add_argument("--loop", help="flexible loop for the 'loop' group: 391-406 (PDB numbering) or new:138-153 (renumbered system, as in gmx make_ndx)")
    a.add_argument("--loop-chain", default="A")
    a.add_argument("--no-site-waters", dest="keep_site_waters", action="store_false",
                   help="drop crystal waters bound to the metal (kept by default)")
    a.add_argument("--site-only", action="store_true", help="stop after the site report")
    a.add_argument("--qm-cores", type=int, default=32)
    a.add_argument("--qm-mem", default="50GB", help="Gaussian %%Mem per job (opt and mk run side by side on cws02: 2 x 50GB fits 128 GB)")
    a.add_argument("--qm-node", default="cws02")
    a.add_argument("--qm-time", default="3-00:00:00")
    a.add_argument("--gaussian-module", default="gaussian/g16-avx2")
    a.add_argument("--mult", type=int, help="spin multiplicity of the QM models (open-shell metals)")
    a.add_argument("--no-large-opt", dest="large_opt", action="store_false",
                   help="MCPB large_opt 0 (default 1, as in the 4LXC protocol)")
    a.add_argument("--no-submit", action="store_true", help="mcpb: write but do not submit the QM jobs")
    a.add_argument("--replicas", type=int, default=10)
    a.add_argument("--temperature", type=float, default=310.0)
    a.add_argument("--md-node", default="cws01", choices=["cws01", "cws02"], help="cws01 = GPU, cws02 = CPU")
    a.add_argument("-o", "--out", required=True)
    a.add_argument("--force", action="store_true", help="rebuild an existing metalprep output directory from scratch")
    f = sub.add_parser("finish", help="(mcpb) MCPB steps 2-4 + system build after the Gaussian jobs")
    f.add_argument("dir")
    c = sub.add_parser("check", help="metal-donor distances vs crystal")
    c.add_argument("dir"); c.add_argument("coords"); c.add_argument("--tpr")
    args = p.parse_args()
    if args.cmd == "prepare":
        args.metal = args.metal or ["ZN:2"]
        args.metal_charge = {}
        for m in args.metal:
            n, q = (m.split(":") + ["2"])[:2]
            args.metal_charge[n.upper()] = int(q)
        args.metal_names = sorted(args.metal_charge)
        zaff_names = {v[2] for v in ZAFF_CENTRES.values()}
        args.metal_names_all = sorted(set(args.metal_names) | (zaff_names if args.model == "zaff" else set()))
        args.loop_new = None
        args.loop_is_new = bool(args.loop and args.loop.startswith("new:"))
        if args.loop:
            lo, hi = args.loop.replace("new:", "").split("-"); args.loop = (int(lo), int(hi))
        if args.model == "nb1264":
            log("note: 12-6-4 systems run with Amber pmemd.cuda / OpenMM only (no GROMACS files)")
        prepare(args)
    elif args.cmd == "finish":
        finish(args)
    else:
        check(args)


if __name__ == "__main__":
    main()
