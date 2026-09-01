import torch
import torch.nn as nn


class AttackClassifier(nn.Module):

    def __init__(self,
                 emb_dim=256,
                 num_classes=15):

        super().__init__()

        self.net = nn.Sequential(

            nn.Linear(emb_dim, 128),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(128, 64),

            nn.ReLU(),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):

        return self.net(x)


if __name__ == "__main__":

    model = AttackClassifier()

    dummy = torch.randn(16, 256)

    out = model(dummy)

    print("=" * 50)
    print("CROSSGUARD Classifier")
    print("=" * 50)
    print("Logits Shape:", out.shape)