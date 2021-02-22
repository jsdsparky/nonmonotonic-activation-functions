# Jack DeLano

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from mylibs.BellCurve import BellCurveFunction
import time
import math

torch.manual_seed(2)

GPU = True
if GPU:
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.fastest = True

HID_DIM = 20
LEARNING_RATE = 0.8
MOMENTUM = 0.65
BATCH_SIZE = 10000

EPOCHS = 3000
EPOCHS_BETWEEN_PRINTS = 10


class LearnedActivationLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(LearnedActivationLayer, self).__init__()
        
        self.lin = nn.Linear(in_dim, out_dim)
        
        self.w_f = nn.Parameter(torch.Tensor(out_dim))
        self.b_f = nn.Parameter(torch.Tensor(out_dim))
        self.w_g = nn.Parameter(torch.Tensor(out_dim))
        self.b_g = nn.Parameter(torch.Tensor(out_dim))
        self.w = nn.Parameter(torch.Tensor(out_dim))
        
        self.lin.weight.data.normal_(0, 0.25)
        self.lin.bias.data.normal_(0, 0.25)
        
        self.w_f.data.normal_(1.5, 0.05) #1.5
        self.w_g.data.normal_(1.75, 0.05) #1.75
        self.b_f.data.normal_(0, 0.05)
        self.b_g.data.normal_(0, 0.05)
        self.w.data.normal_(0.5, 0.025)
    
    def forward(self, x):
        x = self.lin(x)
        
        f = torch.sigmoid(self.w_f*x + self.b_f)
        g = BellCurveFunction.apply(x, self.w_g, self.b_g)
        x = self.w*f + (1 - self.w)*g
        
        return x

class LearnedActivationNN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super(LearnedActivationNN, self).__init__()
        
        self.lal = LearnedActivationLayer(in_dim, hid_dim)
        self.lin1 = nn.Linear(hid_dim, hid_dim)
        self.lin2 = nn.Linear(hid_dim, out_dim)
        
        self.lin1.weight.data.normal_(0, 0.25)
        self.lin2.weight.data.normal_(0, 0.25)
        self.lin1.bias.data.normal_(0, 0.25)
        self.lin2.bias.data.normal_(0, 0.25)
    
    def forward(self, x):
        x = self.lal(x)
        x = torch.sigmoid(self.lin1(x))
        x = torch.sigmoid(self.lin2(x))
        
        return x

# Load data
featuresTrain = np.load('data/xTrain.npy')
featuresTest = np.load('data/xTest.npy')
targetTrain = np.load('data/yTrain.npy')
targetTest = np.load('data/yTest.npy')

# Create tensors from data
XTrain = torch.Tensor(featuresTrain)
YTrain = torch.LongTensor(targetTrain)
XTest = torch.Tensor(featuresTest)
YTest = torch.LongTensor(targetTest)
if GPU:
    XTrain = XTrain.cuda()
    YTrain = YTrain.cuda()
    XTest = XTest.cuda()
    YTest = YTest.cuda()

# Create model
model = LearnedActivationNN(XTrain.size(1), HID_DIM, torch.max(YTrain).item() + 1)
if GPU:
    model.cuda()
lossFunction = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = LEARNING_RATE, momentum = MOMENTUM)

# Train model
timesToSave = []
trainAccsToSave = []
testAccsToSave = []

runningLoss = 0.0
trainCorrect = torch.tensor(0)
testCorrect = torch.tensor(0)
XTrainSizeT = torch.tensor(XTrain.size(0))
XTestSizeT = torch.tensor(XTest.size(0))
if GPU:
    trainCorrect = trainCorrect.cuda()
    testCorrect = testCorrect.cuda()
    XTrainSizeT = XTrainSizeT.cuda()
    XTestSizeT = XTestSizeT.cuda()
timer = time.time()
for i in range(EPOCHS):
    permutation = torch.randperm(XTrain.size(0))
    for j in range(0, XTrain.size(0), BATCH_SIZE):
        dataIndices = permutation[j:j + BATCH_SIZE]
        xVar = XTrain[dataIndices]
        yVar = YTrain[dataIndices].view(-1)
        yHat = model(xVar)
        
        optimizer.zero_grad()
        loss = lossFunction.forward(yHat, yVar)
        loss.backward()
        optimizer.step()
        runningLoss += loss.item()
        
    with torch.no_grad():
        # Update training accuracy
        yHatTrain = model(XTrain)
        trainCorrect += XTrainSizeT - torch.sum(torch.abs(torch.argmax(yHatTrain, dim=1) - YTrain.view(-1)))
        
        # Update test accuracy
        yHatTest = model(XTest)
        testCorrect += XTestSizeT - torch.sum(torch.abs(torch.argmax(yHatTest, dim=1) - YTest.view(-1)))
    
    if i % EPOCHS_BETWEEN_PRINTS == EPOCHS_BETWEEN_PRINTS - 1:
        timeElapsed = time.time() - timer
        trainAcc = 100*trainCorrect.item()/(XTrain.size(0)*EPOCHS_BETWEEN_PRINTS)
        testAcc = 100*testCorrect.item()/(XTest.size(0)*EPOCHS_BETWEEN_PRINTS)
        
        if len(timesToSave) == 0:
            timesToSave.append(timeElapsed)
        else:
            timesToSave.append(timesToSave[-1] + timeElapsed)
        trainAccsToSave.append(trainAcc)
        testAccsToSave.append(testAcc)
        
        
        print("Epoch: {0}, Loss: {1:.4f}, Train Acc: {2:.2f}, Test Acc: {3:.2f},      Time per Epoch: {4: .3f}".format(i+1, runningLoss/EPOCHS_BETWEEN_PRINTS, trainAcc, testAcc, timeElapsed/EPOCHS_BETWEEN_PRINTS))
        runningLoss = 0.0
        trainCorrect -= trainCorrect
        testCorrect -= testCorrect
        timer = time.time()

# Save metrics
np.save('metrics/lal_times.npy', np.array(timesToSave))
np.save('metrics/lal_trainacc.npy', np.array(trainAccsToSave))
np.save('metrics/lal_testacc.npy', np.array(testAccsToSave))

# Test model
yHatTrain = model(XTrain)
trainCorrect = XTrainSizeT - torch.sum(torch.abs(torch.argmax(yHatTrain, dim=1) - YTrain.view(-1)))
print("Final Train Accuracy: " + str(100*trainCorrect.item()/XTrain.size(0)))

yHatTest = model(XTest)
testCorrect = XTestSizeT - torch.sum(torch.abs(torch.argmax(yHatTest, dim=1) - YTest.view(-1)))
print("Final Test Accuracy: " + str(100*testCorrect.item()/XTest.size(0)))

# Plot the learned activation functions
x1 = torch.mm(torch.tensor(np.linspace(-5, 5, 100)).unsqueeze(1), torch.tensor(np.ones(HID_DIM)).unsqueeze(0))
if GPU:
    x1 = x1.cuda()

f = torch.sigmoid(model.lal.w_f*x1 + model.lal.b_f)
#sig_x1_g = torch.sigmoid(model.lal.w_g*x1 + model.lal.b_g)
#g = 4*sig_x1_g*(1-sig_x1_g)
g = BellCurveFunction.apply(x1, model.lal.w_g, model.lal.b_g)
x2 = model.lal.w*f + (1 - model.lal.w)*g

if GPU:
    plt.plot(x1.data.cpu().numpy(), x2.data.cpu().numpy())
else:
    plt.plot(x1.data.numpy(), x2.data.numpy())
plt.show()

