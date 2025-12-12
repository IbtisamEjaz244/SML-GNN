from pymatgen.core import Structure

structure = Structure.from_file("CIFS/1937916.cif")

print(structure)
print("Lattice:", structure.lattice)
print("Number of atoms:", len(structure))
print("Atomic species:", structure.species)
