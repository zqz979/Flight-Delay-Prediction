import utils
import numpy as np
import torch
import random
import torch.nn.functional as F
from torch import nn, optim
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

np.random.seed(0)
random.seed(0)
torch.manual_seed(0)

LR=0.01
EPOCHS=10
TRAIN_BATCH=16
TEST_BATCH=1024

NUM_CLASSES=2

NUM_WORKERS=0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ANN(nn.Module):
    def __init__(self,in_features,num_classes):
        super().__init__()
        self.fc1=nn.Linear(in_features=in_features,out_features=512)
        self.fc2=nn.Linear(in_features=512,out_features=num_classes)
    def forward(self,x):
        pred=F.relu(self.fc1(x))
        pred=self.fc2(pred)
        return pred
    
class SparseDataset(torch.utils.data.Dataset):
    def __init__(self,data,labels):
        self.data=data
        self.labels=labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self,index):
        return self.data[index].toarray()[0],self.labels[index]
    
def train(train_loader, model, optimizer, criterion):
    model=model.train()
    losses = []
    correct=0
    incorrect=0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        correct += torch.sum(output.argmax(axis=1) == target)
        incorrect += torch.sum(output.argmax(axis=1) != target)
    return np.mean(losses), (100.0 * correct / (correct+incorrect))

def test(test_loader, model, criterion):
    model=model.eval()
    losses = []
    correct = 0
    incorrect=0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            losses.append(criterion(output, target).item())
            correct += torch.sum(output.argmax(axis=1) == target)
            incorrect += torch.sum(output.argmax(axis=1) != target)
    return np.mean(losses), (100.0 * correct / (correct+incorrect))

def load_data():
    features,labels=utils.preproc_data(*utils.load_data())
    x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=4)
    train_loader=torch.utils.data.DataLoader(
        SparseDataset(x_train,y_train),
        batch_size=TRAIN_BATCH,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    test_loader=torch.utils.data.DataLoader(
        SparseDataset(x_test,y_test),
        batch_size=TEST_BATCH,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    return train_loader, test_loader, features.shape[1], NUM_CLASSES

def plt_losses(train_losses,test_losses):
    plt.figure()
    plt.plot(range(EPOCHS),train_losses, label="Train Loss")
    plt.plot(range(EPOCHS),test_losses, label="Test Loss")
    plt.title('Train and Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.show()

def main():
    train_loader,test_loader,in_features,num_classes=load_data()
    model=ANN(in_features=in_features,num_classes=num_classes)
    model=model.to(device)
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.SGD(model.parameters(),lr=LR)
    train_losses=[]
    test_losses=[]
    for epoch in range(EPOCHS):
        train_loss,train_acc=train(train_loader,model,optimizer,criterion)
        train_losses.append(train_loss)
        test_loss,test_acc=test(test_loader,model,criterion)
        test_losses.append(test_loss)
        print(f'Epoch: {epoch}, Train Loss: {train_loss}, Test Loss: {test_loss}')
    plt_losses(train_losses,test_losses)
    print(f'Accuracy: {test_acc}')

main()