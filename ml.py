import torch
from torch_geometric.nn import SchNet
from torch_geometric.loader import DataLoader
import pandas as pd
from CIF_to_graph import CrystalDataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.model_selection import train_test_split


# ===========================================================
# 1. Load dataset and create train/val split
# ===========================================================
labels_df = pd.read_csv("labels.csv")
dataset = CrystalDataset("CIFS/", labels_df)

indices = list(range(len(dataset)))
train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

train_dataset = torch.utils.data.Subset(dataset, train_idx)
val_dataset = torch.utils.data.Subset(dataset, val_idx)

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4)


# ===========================================================
# 2. Initialize TensorBoard
# ===========================================================
writer = SummaryWriter("runs/schnet_v2")

writer.add_text("Info", f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")


# ===========================================================
# 3. Define SchNet model
# ===========================================================
model = SchNet(
    hidden_channels=64,
    num_filters=64,
    num_interactions=3,
    cutoff=5.0,
    num_gaussians=25,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
loss_fn = torch.nn.MSELoss()


# ===========================================================
# Helper function: evaluate validation loss
# ===========================================================
def evaluate(loader):
    model.eval()
    total_loss = 0
    preds_all = []
    labels_all = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch.z, batch.pos, batch.batch)
            loss = loss_fn(pred, batch.y)
            total_loss += loss.item() * batch.num_graphs

            preds_all.append(pred.cpu().numpy())
            labels_all.append(batch.y.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    preds_all = np.concatenate(preds_all)
    labels_all = np.concatenate(labels_all)
    return avg_loss, preds_all, labels_all


# ===========================================================
# 4. Training Loop
# ===========================================================
num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch in train_loader:
        batch = batch.to(device)

        optimizer.zero_grad()
        pred = model(batch.z, batch.pos, batch.batch)
        loss = loss_fn(pred, batch.y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.num_graphs

    train_loss = total_loss / len(train_dataset)
    val_loss, preds, labels = evaluate(val_loader)

    # ---- TensorBoard logging ----
    writer.add_scalar("Loss/Train", train_loss, epoch)
    writer.add_scalar("Loss/Validation", val_loss, epoch)
    writer.add_scalar("Learning_rate", scheduler.get_last_lr()[0], epoch)

    # histogram of weights
    for name, param in model.named_parameters():
        writer.add_histogram(f"Weights/{name}", param, epoch)

    # scatter plot: predictions vs labels
    writer.add_scalars("Pred_vs_Label", {
        "mean_pred": float(np.mean(preds)),
        "mean_label": float(np.mean(labels))
    }, epoch)

    print(f"Epoch {epoch+1:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

    scheduler.step()


writer.close()
