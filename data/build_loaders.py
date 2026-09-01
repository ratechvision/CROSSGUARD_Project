from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------
# PATH
# ---------------------------------------------------------

DATA_DIR = Path("data/ids2018")

# Global encoder (same mapping for Train/Val/Test)
LABEL_ENCODER = LabelEncoder()


# ---------------------------------------------------------
# DATASET
# ---------------------------------------------------------

class CICDataset(Dataset):

    def __init__(self, csv_file, fit_encoder=False):

        print(f"Loading {csv_file.name}")

        self.df = pd.read_csv(csv_file, low_memory=False)

        self.df.columns = self.df.columns.str.strip()

        # Remove invalid values
        self.df.replace([np.inf, -np.inf], np.nan, inplace=True)
        self.df.dropna(inplace=True)

        # ---------------- LABELS ----------------

        labels = self.df["Label"]

        if fit_encoder:
            self.labels = LABEL_ENCODER.fit_transform(labels)
        else:
            self.labels = LABEL_ENCODER.transform(labels)

        # ---------------- FEATURES ----------------

        self.features = self.df.drop(columns=["Label"])

        # Keep only numeric columns
        self.features = self.features.select_dtypes(include=np.number)

        # CICIDS2017 uses 78 numeric flow features
        self.features = self.features.iloc[:, :78]

        scaler = StandardScaler()

        self.features = scaler.fit_transform(self.features)

        self.features = torch.tensor(self.features, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        x = self.features[idx]

        # Convert (78,) → (60,78)
        x = x.unsqueeze(0).repeat(60, 1)

        y = self.labels[idx]

        return x, y


# ---------------------------------------------------------
# DATALOADER
# ---------------------------------------------------------

def create_loaders(batch_size=128):

    train = CICDataset(DATA_DIR / "train.csv", fit_encoder=True)

    val = CICDataset(DATA_DIR / "val.csv")

    test = CICDataset(DATA_DIR / "test.csv")

    train_loader = DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


# ---------------------------------------------------------
# TEST
# ---------------------------------------------------------

if __name__ == "__main__":

    train_loader, val_loader, test_loader = create_loaders(32)

    x, y = next(iter(train_loader))

    print("=" * 50)
    print("CROSSGUARD DataLoader")
    print("=" * 50)
    print("Feature Shape :", x.shape)
    print("Label Shape   :", y.shape)
    print("No. Classes   :", len(LABEL_ENCODER.classes_))
    print("\nClasses:")
    for i, c in enumerate(LABEL_ENCODER.classes_):
        print(f"{i:2d} -> {c}")