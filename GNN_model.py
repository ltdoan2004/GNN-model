import torch
from torch.nn import Linear
from torch_geometric.nn import GCNConv

class GNN_model(torch.nn.Module):
    def __init__(self, in_features: int, out_features :int ):
        super(GNN_model, self).__init__()

        self.gcn_1 = GCNConv(in_features, 16)
        self.gcn_2 = GCNConv(16,4)
        self.gcn_3 = GCNConv(4,4)
        self.gcn_4 = GCNConv(4,2)
        self.gcn_5 = GCNConv(2,2)

        self.linear = Linear(2, out_features)

    def forward(self, x, edge_index):
        x = self.gcn_1(x, edge_index)
        x = x.tanh()
        x = self.gcn_2(x, edge_index)
        x = x.tanh()
        x = self.gcn_3(x, edge_index)
        x = x.tanh()
        x = self.gcn_4(x, edge_index)
        x = x.tanh()
        x = self.gcn_5(x, edge_index)
        x = x.tanh()
        output = self.linear(x)

        return output, x
model = GNN_model(4,2)
print(model)
