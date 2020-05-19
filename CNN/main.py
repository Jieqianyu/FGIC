import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
from dataSetProcess import InstantNoodles, Birds
from model import TwoLayerFCNet, ConvNet
import matplotlib.pyplot as plt

# learning parameters
lr = 0.0007
max_epoch = 10

# choose device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# load data
train_root = r'./instant_noodles/train.json'
test_root = r'./instant_noodles/test.json'
train_dataset = InstantNoodles(train_root)
test_dataset = InstantNoodles(test_root)

# train_root = r'./birds/train'
# test_root = r'./birds/test'
# train_dataset = Birds(train_root)
# test_dataset = Birds(test_root)
train_data_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_data_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)

# load module
net = ConvNet(12).to(device)

# choose loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=lr)

# statistics
stats = {}

# train
print('Start training...')
stats['loss_train'] = []
for epoch in range(max_epoch):
    running_loss = 0.0
    for i, (inputs, lables) in enumerate(train_data_loader):
        inputs, lables = inputs.to(device), lables.to(device)
        # clear gradient
        optimizer.zero_grad()
        # forward
        outputs = net(inputs)
        loss = criterion(outputs, lables)
        # backforward
        loss.backward()
        # optimize
        optimizer.step()

        # statistics
        running_loss += loss.item()
        if i % 100 == 99:
            print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss/(i+1)))
    stats['loss_train'].append(running_loss/len(train_dataset))

# test
print('Start testing...')
correct = 0
total = 0
stats['accuracy_test'] = []
for i, (inputs, lables) in enumerate(test_data_loader):
    inputs, lables = inputs.to(device), lables.to(device)
    outputs = net(inputs)
    _, predicted = torch.max(outputs, 1)
    correct += (predicted == lables).sum().item()
    total += lables.size(0)
    stats['accuracy_test'].append(correct/total)

    if i % 100 == 99:
        print('iter = %d, Accuracy: %.1f %%' % (i+1, correct/total*100))
print('Accuracy on test dataset: %.1f %%' % (correct/total*100))  

# Plot
plt.subplot(1, 2, 1)
plt.plot(stats['loss_train'])
plt.title('Loss history')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.plot(stats['accuracy_test'])
plt.title('Accuracy history on test dataset')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.show()
