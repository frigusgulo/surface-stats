from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import pickle
import os
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import numpy as np 

class Dataset(Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, features, labels):
        'Initialization'
        self.labels = labels
        self.features = features

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.labels)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
     
        # Load data and get label
        X = self.features[index,:]
        Y = self.labels[index]

        return X, Y

def unpickle(path):
    with open(path,'rb') as file:
        data = pickle.load(file)
        file.close()
    return data


def is_valid_file(parser, arg):
    if not os.path.isfile(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return arg


class Net(nn.Module):
    def __init__(self,input_size):
        super(Net,self).__init__()
        hidden1 = 512
        hidden2 = 256
        hidden3 = 128

        self.fc1 = nn.Linear(input_size, hidden1)
        self.relu1 = nn.Sigmoid()

        # second hidden layer
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.relu2 = nn.Sigmoid()

        # third hidden layer
        self.fc3 = nn.Linear(hidden2, hidden3)
        self.relu3 = nn.Sigmoid()

        # last output layer
        self.output = nn.Linear(in_features=hidden3,out_features=2) #single output between 0,1



    def forward(self, x):
        '''
        This method takes an input x and layer after layer compute network states.
        Last layer gives us predictions.
        '''
        state = self.fc1(x.float())
        state = self.relu1(state)

        state = self.fc2(state)
        state = self.relu2(state)

        state = self.fc3(state)
        state = self.relu3(state)

        final = self.output(state)
        return final


def train(args, model,criterion, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        onehot = torch.zeros_like(output)
        for i in range(np.max(output.shape)):
            subtarg = torch.zeros_like(output)
            #print(np.int(target[i].item()))
            subtarg[np.int(target[i].item())] = 1
            
        loss = criterion(output,torch.max(subtarg, 1)[1])
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                batch_idx * len(data), 
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item()
                ))


def test(model,criterion, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device).unsqueeze(1)
            output = model(data)
            onehot = torch.zeros_like(output)
            for i in range(np.max(output.shape)):
                subtarg = torch.zeros_like(output)
                #print(np.int(target[i].item()))
                subtarg[np.int(target[i].item())] = 1
            #print(subtarg)
            #print(output)
            test_loss += criterion(output,torch.max(subtarg,1)[1]).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            print(pred)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0e-1, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')

    parser.add_argument('--testset',
                        help='Path to Pickled test data',
                        type=lambda x: is_valid_file(parser, x),
                        dest="testset",
                        required=True,
                        metavar="FILE")
    parser.add_argument('--trainset',
                        help='Path to Pickled train data',
                        type=lambda x: is_valid_file(parser, x),
                        dest="trainset",
                        required=True,
                        metavar="FILE")

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_data = unpickle(args.trainset)

    print("Training Data Length: {}\n".format(len(train_data)))
    train_loader = torch.utils.data.DataLoader(
        dataset = train_data,
        batch_size=args.batch_size, 
        shuffle=True,
        **kwargs)
    

    test_data = unpickle(args.testset)

    print("Test Data Length: {}\n".format(len(test_data)))
    test_loader = torch.utils.data.DataLoader(
        dataset = test_data, 
        batch_size=args.test_batch_size, 
        shuffle=True,
        **kwargs)

    model = Net(25).to(device)
 
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(args, model,criterion, device, train_loader, optimizer, epoch)
        test(model,criterion,device, test_loader)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), "/home/fdunbar/Research/surface-stats/msgl_nn.pt")


if __name__ == '__main__':
    main()