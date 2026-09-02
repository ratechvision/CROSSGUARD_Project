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

# Global Label Encoder
LABEL_ENCODER = LabelEncoder()

# Global Feature Scaler
SCALER = StandardScaler()


# ---------------------------------------------------------
# DATASET
# ---------------------------------------------------------

class CICDataset(Dataset):

    def __init__(self, csv_file, fit=False):

        print(f"Loading {csv_file.name}")

        df = pd.read_csv(csv_file, low_memory=False)
        df.columns = df.columns.str.strip()

        # Remove invalid values
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)

        # ---------------- LABEL ----------------

        labels = df["Label"]

        if fit:
            self.labels = LABEL_ENCODER.fit_transform(labels)
        else:
            self.labels = LABEL_ENCODER.transform(labels)

        # ---------------- FEATURES ----------------

        features = df.drop(columns=["Label"])

        # Keep only numeric features
        features = features.select_dtypes(include=np.number)

        # CICIDS2017 = 78 numerical flow features
        features = features.iloc[:, :78]

        # Use ONE scaler for all splits
        if fit:
            features = SCALER.fit_transform(features)
        else:
            features = SCALER.transform(features)

        self.features = torch.tensor(features, dtype=torch.float32)
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

def create_loaders(batch_size=256):

    train_dataset = CICDataset(DATA_DIR / "train.csv", fit=True)

    val_dataset = CICDataset(DATA_DIR / "val.csv")

    test_dataset = CICDataset(DATA_DIR / "test.csv")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
    )

    return train_loader, val_loader, test_loader


# ---------------------------------------------------------
# UNIT TEST
# ---------------------------------------------------------

if __name__ == "__main__":

    train_loader, val_loader, test_loader = create_loaders()

    x, y = next(iter(train_loader))

    print("=" * 50)
    print("CROSSGUARD DataLoader")
    print("=" * 50)
    print("Feature Shape :", x.shape)
    print("Label Shape   :", y.shape)
    print("No. Classes   :", len(LABEL_ENCODER.classes_))

    print("\nClasses:")
    for i, cls in enumerate(LABEL_ENCODER.classes_):
        print(f"{i:2d} -> {cls}")
