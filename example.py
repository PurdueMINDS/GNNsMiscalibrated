"""Simple example usage of the Friendster dataset.

Most of the code shown here is copied from PyTorch Geometric's example for GCN:
https://github.com/rusty1s/pytorch_geometric/blob/master/examples/gcn.py

The purpose of this file is to illustrate how to access the data and use the
provided Friendster class.
"""

import os.path as osp
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from friendster import Friendster

class Standardize(object):
    """Standardize the columns of data.x to have mean 0 and variance 1"""
    def __call__(self, data):
        x = data.x
        mu = x.mean(0)
        sigma = x.std(0)
        sigma[sigma == 0] = 1

        data.x = (x - mu) / sigma

        return data

# Copied mostly from PyTorch Geometric's example for GCN:
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, 256)
        self.conv2 = GCNConv(256, n_classes)  # Ignore the NA class

    def forward(self):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

print("Loading dataset...")
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', "Friendster")
transform = Standardize()
dataset = Friendster(path, transform=transform)
n_classes = 4  # The 5th class is NA, which we ignore
data = dataset[0]  # There's only one graph in the dataset

print("Preparing model and optimizer...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = Net().to(device), data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

print("Computing class weights...")
train_weights = torch.bincount(data.y[data.validation_mask]).float()
train_weights = train_weights.max() / train_weights
train_weights = train_weights.to(device)

def train():
    model.train()
    optimizer.zero_grad()
    F.nll_loss(model()[data.train_mask], data.y[data.train_mask], weight=train_weights).backward()
    optimizer.step()

def test():
    model.eval()
    logits, accs = model(), []
    for _, mask in data('train_mask', 'validation_mask', 'test_mask'):
        # Compute the weights for each class
        weights = torch.bincount(data.y[mask]).float()
        weights = weights.max() / weights
        # Get the associated weights for each example
        weights = weights[data.y[mask]]
        # Compute (balanced) accuracy
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).float().mul(weights).sum().item() / weights.sum().item()
        accs.append(acc)
    return accs

print("Beginning training...")
best_val_acc = test_acc = 0
best_epoch = 0
for epoch in range(1, 201):
    train()
    train_acc, val_acc, tmp_test_acc = test()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_acc = tmp_test_acc
        best_epoch = epoch
    log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f} | Best Val: {:.4f} Test: {:.4f} (from epoch {:d})'
    print(log.format(epoch, train_acc, val_acc, tmp_test_acc, best_val_acc, test_acc, best_epoch))

