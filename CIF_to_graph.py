import torch
import glob
import numpy as np
from pymatgen.core import Structure
from torch_geometric.data import Data, Dataset


# ---------------------------------------------------------
# Convert CIF file to a PyTorch Geometric graph
# ---------------------------------------------------------
def cif_to_graph(cif_path, cutoff=5.0):

    structure = Structure.from_file(cif_path)

    # atomic numbers (SchNet uses 'z')
    z = torch.tensor([site.specie.Z for site in structure], dtype=torch.long)

    # Cartesian positions (SchNet uses 'pos')
    pos = torch.tensor(structure.cart_coords, dtype=torch.float)

    # Build edges using neighbor list
    edge_index = []
    for i, site_i in enumerate(structure):
        neighbors = structure.get_neighbors(site_i, cutoff)
        for n in neighbors:
            j = n.index
            edge_index.append([i, j])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    # Package into Data object for SchNet
    return Data(
        z=z,         # atomic numbers
        pos=pos,     # Cartesian coordinates
        edge_index=edge_index
    )


# ---------------------------------------------------------
# Dataset class that maps CIF -> graph + label
# ---------------------------------------------------------
class CrystalDataset(Dataset):
    def __init__(self, cif_dir, labels_df, transform=None):
        super().__init__(root=None, transform=transform)

        # All CIF files from directory
        self.cif_files = sorted(glob.glob(f"{cif_dir}/*.cif"))

        # Index labels by filename only
        labels_df = labels_df.set_index("filename")
        self.labels_df = labels_df

    def len(self):
        return len(self.cif_files)

    def get(self, idx):
        path = self.cif_files[idx]
        fname = path.split("/")[-1]      # extract crystalNN.cif

        # Build graph
        graph = cif_to_graph(path)

        # Get correct label
        label = self.labels_df.loc[fname]["response_time"]
        graph.y = torch.tensor([label], dtype=torch.float)

        return graph
