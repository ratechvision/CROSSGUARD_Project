import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import HANConv


class GraphEncoder(nn.Module):

    def __init__(self,
                 input_dim=16,
                 hidden_dim=64,
                 emb_dim=256):

        super().__init__()

        metadata = (
            ["host", "ip", "port"],
            [
                ("host", "connects", "ip"),
                ("ip", "rev_connects", "host"),
                ("ip", "uses", "port"),
                ("port", "rev_uses", "ip"),
            ],
        )

        self.han = HANConv(
            in_channels={
                "host": input_dim,
                "ip": input_dim,
                "port": input_dim,
            },
            out_channels=hidden_dim,
            metadata=metadata,
            heads=4,
        )

        # HAN output is 64-dimensional
        self.project = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, emb_dim)
        )

    def forward(self, data):

        x_dict = self.han(data.x_dict, data.edge_index_dict)

        host = x_dict["host"]

        graph_embedding = host.mean(dim=0, keepdim=True)

        return self.project(graph_embedding)


def build_demo_graph():

    data = HeteroData()

    data["host"].x = torch.randn(10, 16)
    data["ip"].x = torch.randn(15, 16)
    data["port"].x = torch.randn(8, 16)

    src = torch.randint(0, 10, (30,))
    dst = torch.randint(0, 15, (30,))
    data["host", "connects", "ip"].edge_index = torch.stack([src, dst])

    data["ip", "rev_connects", "host"].edge_index = torch.stack([dst, src])

    src2 = torch.randint(0, 15, (25,))
    dst2 = torch.randint(0, 8, (25,))
    data["ip", "uses", "port"].edge_index = torch.stack([src2, dst2])

    data["port", "rev_uses", "ip"].edge_index = torch.stack([dst2, src2])

    return data


if __name__ == "__main__":

    graph = build_demo_graph()

    model = GraphEncoder()

    emb = model(graph)

    print("=" * 50)
    print("CROSSGUARD Graph Encoder")
    print("=" * 50)
    print("Embedding Shape:", emb.shape)