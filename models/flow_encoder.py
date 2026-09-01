import torch
import torch.nn as nn


# ---------------------------------------------------------
# CHOMP LAYER
# ---------------------------------------------------------
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size]


# ---------------------------------------------------------
# TEMPORAL RESIDUAL BLOCK
# ---------------------------------------------------------
class TemporalBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        dilation=1,
        dropout=0.2
    ):
        super().__init__()

        padding = (kernel_size - 1) * dilation

        self.net = nn.Sequential(

            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation
            ),

            Chomp1d(padding),

            nn.BatchNorm1d(out_channels),

            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation
            ),

            Chomp1d(padding),

            nn.BatchNorm1d(out_channels),

            nn.ReLU(),

            nn.Dropout(dropout)
        )

        if in_channels != out_channels:
            self.downsample = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        residual = self.downsample(x)
        output = self.net(x)
        return torch.relu(output + residual)


# ---------------------------------------------------------
# FLOW ENCODER (TCN)
# ---------------------------------------------------------
class FlowEncoder(nn.Module):

    def __init__(
        self,
        input_features=78,
        embedding_dim=256,
        dropout=0.2
    ):

        super().__init__()

        self.tcn = nn.Sequential(

            TemporalBlock(
                input_features,
                64,
                dilation=1,
                dropout=dropout
            ),

            TemporalBlock(
                64,
                128,
                dilation=2,
                dropout=dropout
            ),

            TemporalBlock(
                128,
                embedding_dim,
                dilation=4,
                dropout=dropout
            )
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.projection = nn.Sequential(

            nn.Linear(embedding_dim, embedding_dim),

            nn.ReLU(),

            nn.Dropout(dropout)
        )

    def forward(self, x):

        # Input : (Batch, 60, 78)

        x = x.transpose(1, 2)

        x = self.tcn(x)

        x = self.global_pool(x).squeeze(-1)

        x = self.projection(x)

        return x


# ---------------------------------------------------------
# UNIT TEST
# ---------------------------------------------------------
if __name__ == "__main__":

    model = FlowEncoder()

    dummy = torch.randn(8, 60, 78)

    output = model(dummy)

    print("=" * 50)
    print("CROSSGUARD Flow Encoder")
    print("=" * 50)
    print("Input Shape :", dummy.shape)
    print("Output Shape:", output.shape)