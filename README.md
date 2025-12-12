# 🧬 SML-GNN: Crystal Graph Neural Network Framework

SML-GNN is a modular machine learning pipeline designed to train Graph Neural Networks (GNNs) on crystalline materials represented in CIF format.  
The project converts CIF structures into graph data, processes them using PyTorch Geometric, and trains a SchNet model to perform regression on material properties.

This repository is meant to serve as a clean, extendable template for crystal-based ML research.

---

## 🚀 Project Overview

Traditional machine learning struggles with crystalline structures because materials are **irregular**, **3D**, and **non-Euclidean**.  
Graph Neural Networks (GNNs) solve this by representing crystals as graphs:

- **Nodes** = atoms  
- **Edges** = spatial relationships (distances / neighbors)  
- **Node features** = atomic numbers  
- **Edge features** = geometric distances (used internally by SchNet)

SML-GNN provides:

- A robust CIF → PyG graph converter using `pymatgen`  
- A PyTorch Geometric Dataset class for loading data  
- A full SchNet architecture for property prediction  
- TensorBoard integration for experiment tracking  
- A clean, reproducible training pipeline  

This repository is ideal for:

- property prediction  
- GNN model experimentation  
- materials informatics education  
- benchmarking new architectures on crystal data  

---

## Key Features

- **Automatic CIF parsing and graph construction**  
- **SchNet model with Gaussian distance filters**  
- **TensorBoard logging for training curves + weight histograms**  
- **Lightweight, minimal dataset interface**  
- **Simple test script to verify CIF loading**  
- **Modular file structure for easy modification**  

---

## Requirements

Install all dependencies from:


Key libraries include:

- `pytorch`
- `torch-geometric`
- `pymatgen`
- `numpy`, `pandas`, `scikit-learn`
- `tensorboard`

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/SML-GNN.git
cd SML-GNN

python3 -m venv .venv
source .venv/bin/activate
.\.venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```
If pip ever complains about PyG wheels, install PyG manually:
```
pip install torch-geometric
```
Then do pip install requirements again.

## 1. Add CIF files

Place your CIF files inside:

SML-GNN/CIFS/
Example:
```
CIFS/
    material1.cif
    material2.cif
    material3.cif
```

Note: CIFs are intentionally NOT included in the repo.
Users must add their own dataset.

## 2. Create labels.csv

In the project root:
```
filename,response_time
material1.cif,0.91
material2.cif,1.22
material3.cif,0.53
```
filename must match your CIF filenames exactly

response_time can be any numeric target (rename as needed)

## Testing CIF Loading

Before training, confirm installation using:
```bash
python test.py
```

You should see output similar to:
```
Dataset length: 10
Data(x=[36, 1], edge_index=[2, 956], ...)
tensor([1.5088])
```

If this runs successfully, your dataset is configured correctly.

## Training the SchNet GNN Model

Run:
```
python ml.py
```
This script:

- loads CIFs
- converts them into graph objects
- splits dataset into train/validation
- trains a SchNet model
- logs training + validation loss
- writes TensorBoard logs
- saves training logs to train_logs_schnet.txt

Example Output:
```
Epoch 001 | Train Loss: 0.9123 | Val Loss: 0.8841
Epoch 002 | Train Loss: 0.6532 | Val Loss: 0.7021
...
```

## Visualizing Training with TensorBoard

Launch TensorBoard:
```
tensorboard --logdir=runs
```

Then open your browser:
http://localhost:6006

You can now inspect:
- Training loss
- Validation loss
- Learning rate
- Weight histograms
- Prediction statistics
- Run metadata

## Project Structure
```
SML-GNN/
│
├── CIF_to_graph.py          # CIF → Graph conversion
├── train_schnet.py          # Main SchNet training script
├── test.py                  # Dataset loader test
├── labels.csv               # User-provided labels
│
├── CIFS/                    # Place your CIF files here
│   └── .gitkeep
│
├── runs/                    # TensorBoard logs (ignored in git)
└── train_logs_schnet.txt    # Plain-text log output
```
