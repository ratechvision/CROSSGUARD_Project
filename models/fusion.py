import torch
import torch.nn as nn


class GatedCrossAttentionFusion(nn.Module):

    def __init__(self,
                 emb_dim=256,
                 heads=8,
                 dropout=0.2):

        super().__init__()

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=emb_dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True
        )

        self.gate = nn.Sequential(
            nn.Linear(emb_dim * 3, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
            nn.Softmax(dim=-1)
        )

        self.output = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, flow, text, graph):

        # flow,text : (B,256)
        # graph     : (1,256)

        if graph.shape[0] == 1:
            graph = graph.repeat(flow.shape[0], 1)

        tokens = torch.stack([flow, text, graph], dim=1)

        attended, _ = self.cross_attention(
            tokens,
            tokens,
            tokens
        )

        concat = torch.cat([flow, text, graph], dim=1)

        weights = self.gate(concat)

        fused = (
            weights[:, 0:1] * attended[:, 0] +
            weights[:, 1:2] * attended[:, 1] +
            weights[:, 2:3] * attended[:, 2]
        )

        return self.output(fused), weights


if __name__ == "__main__":

    B = 8

    flow = torch.randn(B, 256)
    text = torch.randn(B, 256)
    graph = torch.randn(1, 256)

    model = GatedCrossAttentionFusion()

    fused, gates = model(flow, text, graph)

    print("=" * 50)
    print("CROSSGUARD Fusion Module")
    print("=" * 50)
    print("Fused Shape :", fused.shape)
    print("Gate Shape  :", gates.shape)
    print("Example Gate:", gates[0])