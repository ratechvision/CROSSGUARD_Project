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

EPOCHS = 5
LR = 1e-4
NUM_CLASSES = 15

print(f"[CROSSGUARD] Device : {DEVICE}")

# -------------------------------------------------------
# DATA
# -------------------------------------------------------

train_loader, val_loader, test_loader = create_loaders(batch_size=64)

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

history = []


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

    for step, (x, y) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        flow_emb = flow_encoder(x)

        text_emb = text_encoder(batch_reports(len(x), step))

        graph_emb = graph_encoder(graph)

        fused, gates = fusion(flow_emb, text_emb, graph_emb)

        logits = classifier(fused)

        loss = criterion(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    history.append([epoch + 1, avg_loss])

    print(f"Epoch {epoch+1} Loss = {avg_loss:.4f}")

# -------------------------------------------------------
# TEST
# -------------------------------------------------------

print("\nEvaluating...")

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
    columns=["epoch", "train_loss"]
).to_csv(ART / "train_log.csv", index=False)

np.save(ART / "y_true.npy", np.array(y_true))
np.save(ART / "y_pred.npy", np.array(y_pred))
np.save(ART / "y_prob.npy", np.array(y_prob))
np.save(ART / "gates.npy", np.array(gate_store))

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

print("\n==============================")
print("CROSSGUARD TEST RESULTS")
print("==============================")
print(f"Accuracy        : {acc:.4f}")
print(f"Macro Precision : {p:.4f}")
print(f"Macro Recall    : {r:.4f}")
print(f"Macro F1        : {f1:.4f}")
print("\nArtifacts saved to ./artifacts/")