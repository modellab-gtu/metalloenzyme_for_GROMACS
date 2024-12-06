import numpy as np
import argparse

# Function to randomly displace atomic positions
def random_displacement(position, max_variation=0.1):
    displacement = np.random.uniform(-max_variation, max_variation, size=3)
    return position + displacement

# Function to read an XYZ file and get atomic positions
def read_xyz(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        atom_count = int(lines[0].strip())
        atoms = [line.split()[0] for line in lines[2:2+atom_count]]
        positions = np.array([[float(x) for x in line.split()[1:4]] for line in lines[2:2+atom_count]])
    return atoms, positions

# Function to write an XYZ file
def write_xyz(file_path, atoms, positions):
    with open(file_path, 'w') as f:
        f.write(f"{len(atoms)}\n")
        f.write("Randomly displaced structure\n")
        for atom, pos in zip(atoms, positions):
            f.write(f"{atom} {pos[0]:.5f} {pos[1]:.5f} {pos[2]:.5f}\n")

# Function to calculate RMSD between initial and final structures for heavy atoms
def calculate_rmsd(initial_positions, final_positions, atom_types):
    # Filter out hydrogen atoms (assuming 'H' for hydrogen)
    heavy_atom_mask = [atom != 'H' for atom in atom_types]
    initial_positions_heavy = initial_positions[heavy_atom_mask]
    final_positions_heavy = final_positions[heavy_atom_mask]
    
    # Compute the RMSD
    diff = final_positions_heavy - initial_positions_heavy
    rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
    return rmsd

# Main function to apply random displacements, check RMSD, and write the output
def apply_random_displacements(input_file, output_file, max_variation, rmsd_min, rmsd_max, num_conf):
    atoms, initial_positions = read_xyz(input_file)
    output = output_file.rsplit('.', 1)[0]
    disstrnum = 1
    # Generate random displacements and calculate RMSD until within range
    while True:
        final_positions = np.array([random_displacement(pos, max_variation) for pos in initial_positions])
        rmsd = calculate_rmsd(initial_positions, final_positions, atoms)
        
        if rmsd_min <= rmsd <= rmsd_max:
            print(f"Accepted structure with RMSD: {rmsd:.3f} Å")
            write_xyz(f"{output}{disstrnum}.xyz", atoms, final_positions)
            disstrnum += 1
            if disstrnum > num_conf:
                break
        else:
            print(f"RMSD {rmsd:.3f} Å out of range. Retrying...")

# Script entry point
def main():
    parser = argparse.ArgumentParser(description="Randomly displace atomic positions in an XYZ file and accept the change if RMSD is within a specified range.")
    parser.add_argument("-input", type=str, required=True, help="Input XYZ file")
    parser.add_argument("-output", type=str, required=True, help="Output XYZ file")
    parser.add_argument("-max_variation", type=float, default=0.1, help="Maximum variation for random displacement")
    parser.add_argument("-rmsd_min", type=float, default=0.5, help="Minimum acceptable RMSD (Å)")
    parser.add_argument("-rmsd_max", type=float, default=2.0, help="Maximum acceptable RMSD (Å)")
    parser.add_argument("-num_conf", type=int, default=1, help="Number of distorted sructures)")
    args = parser.parse_args()

    apply_random_displacements(args.input, args.output, args.max_variation, args.rmsd_min, args.rmsd_max, args.num_conf)

if __name__ == "__main__":
    main()

