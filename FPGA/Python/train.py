import torch
import torch.nn as nn
from pathlib import Path
from dataset import FootDropDataset
from model import FootDropCNN
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score

DATA_DIR    = str(Path(__file__).resolve().parent / "enabl3s_dataset_sliding")
VAL_SUBJECT = "AB185"
EPOCHS      = 25
LR          = 1e-3
BATCH_SIZE  = 64
subjects = ("AB185", "AB188", "AB186", "AB156", "AB189", "AB190", "AB191", "AB192", "AB193", "AB194")

C1           = 16
C2           = 32
LSTM_HIDDEN  = 32
DROPOUT      = 0.3
WEIGHT_DECAY = 1e-4

train_ds = FootDropDataset(DATA_DIR, (VAL_SUBJECT,))
val_ds = FootDropDataset(DATA_DIR, tuple(s for s in subjects if s != VAL_SUBJECT))

torch.manual_seed(123)

train_loader = DataLoader(
    dataset=train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    drop_last=True
)

val_loader = DataLoader(
    dataset=val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

num_pos = (train_ds.labels == 1).sum().item()
num_neg = (train_ds.labels == 0).sum().item()
pos_weight = torch.tensor([num_neg / num_pos])

print(f"Positives: {num_pos}, Negatives: {num_neg}, pos_weight: {pos_weight.item():.2f}")

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

model = FootDropCNN(c1=C1, c2=C2, lstm_hidden=LSTM_HIDDEN, dropout=DROPOUT)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

best_f1 = 0.0

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        logits = model(inputs).squeeze(1)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        val_loss = 0.0
        for val_inputs, val_labels in val_loader:
            logits = model(val_inputs).squeeze(1)
            val_loss += criterion(logits, val_labels).item()

            preds = (logits > 0).int()
            all_preds.append(preds)
            all_labels.append(val_labels.int())

    avg_val_loss = val_loss / len(val_loader)

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    val_f1 = f1_score(all_labels, all_preds)
    val_precision = precision_score(all_labels, all_preds)
    val_recall = recall_score(all_labels, all_preds)

    print(f"Epoch {epoch+1}/{EPOCHS}  "
          f"Train loss: {avg_train_loss:.4f}  "
          f"Val loss: {avg_val_loss:.4f}  "
          f"F1: {val_f1:.3f}  "
          f"Precision: {val_precision:.3f}  "
          f"Recall: {val_recall:.3f}")

    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model.state_dict(), 'best_model.pt')
        print(f"  → New best model saved (F1: {best_f1:.3f})")

print("Done. Best F1:", best_f1)
