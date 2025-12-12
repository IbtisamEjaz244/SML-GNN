import pandas as pd
from CIF_to_graph import CrystalDataset

labels_df = pd.read_csv("labels.csv")
dataset = CrystalDataset("CIFS/", labels_df)

print("Dataset length:", len(dataset))

sample = dataset[0]
print(sample)
print(sample.x.shape)
print(sample.edge_index.shape)
print(sample.y)
