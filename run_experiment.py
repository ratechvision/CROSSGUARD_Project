import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

from data.build_loaders import create_loaders, LABEL_ENCODER
from models.flow_encoder import FlowEncoder
from models.text_encoder import TextEncoder, load_reports
from models.graph_encoder import GraphEncoder, build_demo_graph
from models.fusion import GatedCrossAttentionFusion
from models.classifier import AttackClassifier

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------

ART = Path("artifacts")
ART.mkdir(exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EPOCHS = 30
BATCH_SIZE = 256
LR = 2e-4
NUM_CLASSES = 15
PATIENCE = 5

print("=" * 50)
print("CROSSGUARD TRAINING")
print("=" * 50)
print("Device :", DEVICE)

# -------------------------------------------------------
# DATA
# -------------------------------------------------------

train_loader, val_loader, test_loader = create_loaders(batch_size=BATCH_SIZE)

reports = load_reports()
CLASS_NAMES = list(LABEL_ENCODER.classes_)
graph = build_demo_graph().to(DEVICE)

# -------------------------------------------------------
# MODEL
# -------------------------------------------------------

flow_encoder = FlowEncoder().to(DEVICE)
text_encoder = TextEncoder().to(DEVICE)
graph_encoder = GraphEncoder().to(DEVICE)
fusion = GatedCrossAttentionFusion().to(DEVICE)
classifier = AttackClassifier(num_classes=NUM_CLASSES).to(DEVICE)

params = (
    list(flow_encoder.parameters())
    + list(text_encoder.parameters())
    + list(graph_encoder.parameters())
    + list(fusion.parameters())
    + list(classifier.parameters())
)

optimizer = torch.optim.AdamW(params, lr=LR)
criterion = nn.CrossEntropyLoss()

scaler = torch.amp.GradScaler("cuda")

history = []
best_f1 = 0
patience_counter = 0

# -------------------------------------------------------
# HELPER
# -------------------------------------------------------

def batch_reports(batch_size, offset):
    txt = []
    for i in range(batch_size):
        txt.append(reports[(offset + i) % len(reports)])
    return txt

# -------------------------------------------------------
# TRAIN
# -------------------------------------------------------

for epoch in range(EPOCHS):

    flow_encoder.train()
    text_encoder.train()
    graph_encoder.train()
    fusion.train()
    classifier.train()

    total_loss = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

    for step, (x, y) in enumerate(loop):

        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):

            flow_emb = flow_encoder(x)
            text_emb = text_encoder(batch_reports(len(x), step))
            graph_emb = graph_encoder(graph)

            fused, _ = fusion(flow_emb, text_emb, graph_emb)
            logits = classifier(fused)

            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        loop.set_postfix(loss=f"{loss.item():.4f}")

    train_loss = total_loss / len(train_loader)

    # ---------------------------------------------------
    # VALIDATION
    # ---------------------------------------------------

    flow_encoder.eval()
    text_encoder.eval()
    graph_encoder.eval()
    fusion.eval()
    classifier.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():

        graph_emb = graph_encoder(graph)

        for step, (x, y) in enumerate(val_loader):

            x = x.to(DEVICE)

            with torch.amp.autocast("cuda"):

                flow_emb = flow_encoder(x)
                text_emb = text_encoder(batch_reports(len(x), step))
                fused, _ = fusion(flow_emb, text_emb, graph_emb)
                logits = classifier(fused)

            pred = logits.argmax(1)

            y_true.extend(y.numpy())
            y_pred.extend(pred.cpu().numpy())

    _, _, val_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    history.append([epoch + 1, train_loss, val_f1])

    print(
        f"Epoch {epoch+1} | "
        f"Loss={train_loss:.4f} | "
        f"Val F1={val_f1:.4f}"
    )

    # Save Best Model

    if val_f1 > best_f1:

        best_f1 = val_f1
        patience_counter = 0

        torch.save(
            {
                "flow": flow_encoder.state_dict(),
                "text": text_encoder.state_dict(),
                "graph": graph_encoder.state_dict(),
                "fusion": fusion.state_dict(),
                "classifier": classifier.state_dict(),
            },
            ART / "best.pt",
        )

        print("✓ Best model saved")

    else:

        patience_counter += 1

    if patience_counter >= PATIENCE:

        print("Early stopping triggered.")
        break

# -------------------------------------------------------
# LOAD BEST MODEL
# -------------------------------------------------------

ckpt = torch.load(ART / "best.pt", map_location=DEVICE)

flow_encoder.load_state_dict(ckpt["flow"])
text_encoder.load_state_dict(ckpt["text"])
graph_encoder.load_state_dict(ckpt["graph"])
fusion.load_state_dict(ckpt["fusion"])
classifier.load_state_dict(ckpt["classifier"])

# -------------------------------------------------------
# TEST
# -------------------------------------------------------

print("\nEvaluating Best Model...")

flow_encoder.eval()
text_encoder.eval()
graph_encoder.eval()
fusion.eval()
classifier.eval()

y_true = []
y_pred = []
y_prob = []
gate_store = []

with torch.no_grad():

    graph_emb = graph_encoder(graph)

    for step, (x, y) in enumerate(tqdm(test_loader)):

        x = x.to(DEVICE)

        with torch.amp.autocast("cuda"):

            flow_emb = flow_encoder(x)
            text_emb = text_encoder(batch_reports(len(x), step))
            fused, gates = fusion(flow_emb, text_emb, graph_emb)
            logits = classifier(fused)

        prob = torch.softmax(logits, dim=1)
        pred = prob.argmax(1)

        y_true.extend(y.numpy())
        y_pred.extend(pred.cpu().numpy())
        y_prob.extend(prob.cpu().numpy())
        gate_store.extend(gates.cpu().numpy())

acc = accuracy_score(y_true, y_pred)

p, r, f1, _ = precision_recall_fscore_support(
    y_true,
    y_pred,
    average="macro",
    zero_division=0,
)

metrics = {
    "accuracy": float(acc),
    "macro_precision": float(p),
    "macro_recall": float(r),
    "macro_f1": float(f1),
    "classes": CLASS_NAMES,
}

with open(ART / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

pd.DataFrame(
    history,
    columns=["epoch", "train_loss", "val_f1"],
).to_csv(ART / "train_log.csv", index=False)

np.save(ART / "y_true.npy", np.array(y_true))
np.save(ART / "y_pred.npy", np.array(y_pred))
np.save(ART / "y_prob.npy", np.array(y_prob))
np.save(ART / "gates.npy", np.array(gate_store))

print("\n" + "=" * 50)
print("CROSSGUARD FINAL TEST RESULTS")
print("=" * 50)
print(f"Accuracy        : {acc:.4f}")
print(f"Macro Precision : {p:.4f}")
print(f"Macro Recall    : {r:.4f}")
print(f"Macro F1        : {f1:.4f}")
print("=" * 50)
print("Artifacts saved in ./artifacts/")
