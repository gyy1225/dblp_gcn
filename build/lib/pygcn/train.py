from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import visdom
import torch
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from pygcn.utils import load_data, f1
from pygcn.models import GCN
from pygcn.pytorchrtools import EarlyStopping
from visdom import Visdom
import numpy as np
import time
from sklearn.metrics import classification_report

early_stopping = EarlyStopping(patience=20, verbose=False)

train_loss_list = []
train_f1_list = []
val_loss_list=[]
val_f1_list=[]
target_names=["0","1","2","3"]
# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=240,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=10,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
adj, features, labels, idx_train, idx_val, idx_test = load_data()

# Model and optimizer
model = GCN(nfeat=features.shape[1],
            nhid=args.hidden,
            nclass=labels.max().item() + 1,
            dropout=args.dropout)
optimizer = optim.Adam(model.parameters(),
                       lr=args.lr, weight_decay=args.weight_decay)
# optimizer = optim.Adam(model.parameters(),
#                        lr=args.lr, weight_decay=10)                       

if args.cuda:
    model.cuda()
    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    f1_train = f1(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately,
        # deactivates dropout during validation run.
        model.eval()
        output = model(features, adj)
    
    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    f1_val = f1(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'f1_train: {:.4f}'.format(f1_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'f1_val: {:.4f}'.format(f1_val.item()),
          'time: {:.4f}s'.format(time.time() - t))
    train_loss_list.append(loss_train)
    train_f1_list.append(f1_train)
    val_loss_list.append(loss_val)
    val_f1_list.append(f1_val)
    #print(output)
    preds = output[idx_train].max(1)[1].type_as(labels[idx_train])
    #print(preds,labels[idx_val])
    report = classification_report(labels[idx_train],preds)
    print(report)
    return loss_train,f1_train,loss_val,f1_val

def test():
    model.eval()
    output = model(features, adj)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    f1_test = f1(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "f1= {:.4f}".format(f1_test.item()))
    preds = output[idx_test].max(1)[1].type_as(labels[idx_test])
    report = classification_report(labels[idx_test],preds)
    print('Test classification report:\n',
                report)

# Train model
t_total = time.time()
for epoch in range(args.epochs):
    loss_train,f1_train,loss_val,f1_val=train(epoch)
    early_stopping(loss_val, model)
   
print("Optimization Finished!")


x1 = range(0, args.epochs)
x2 = range(0,args.epochs )
y1 = train_f1_list
y2 = train_loss_list
y3=val_f1_list
y4=val_loss_list
plt.subplot(2, 2, 1)
plt.plot(x1, y1, 'o-')
plt.title('Train f1 vs. epoches')
plt.ylabel('Train f1')
plt.subplot(2, 2, 2)
plt.plot(x2, y2, '.-')
plt.xlabel('Train loss vs. epoches')
plt.ylabel('Train loss')
plt.subplot(2, 2, 3)
plt.plot(x1, y3, 'o-')
plt.title('Val f1 vs. epoches')
plt.ylabel('Val f1')
plt.subplot(2, 2, 4)
plt.plot(x2, y4, '.-')
plt.xlabel('Val loss vs. epoches')
plt.ylabel('Val loss')

#plt.savefig("accuracy_loss.jpg")

print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()
plt.show()