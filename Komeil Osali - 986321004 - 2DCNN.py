#Komeil Osali - https://github.com/komeilosali
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

# Define the CNN2D model
class CNN2D(nn.Module):
    def __init__(self):
        super(CNN2D, self).__init__()

        #kernel size = 3 , stride=1, padding=1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        #kernel size = 5 , stride=1, padding=2
        #self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        #self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)

        # kernel size = 7 , stride=1, padding=3
        #self.conv1 = nn.Conv2d(1, 32, kernel_size=7, stride=1, padding=3)
        #self.conv2 = nn.Conv2d(32, 64, kernel_size=7, stride=1, padding=3)

        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, 10)



    def forward(self, x):

        #tan h
        #x = self.pool(torch.tanh(self.conv1(x)))
        #x = self.pool(torch.tanh(self.conv2(x)))

        #Sigmoid
        #x = self.pool(torch.sigmoid(self.conv1(x)))
        #x = self.pool(torch.sigmoid(self.conv2(x)))

        # relu
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))

        x = x.view(-1, 64*7*7)

        #x = torch.tanh(self.fc1(x))
        #x = torch.sigmoid(self.fc1(x))
        x = torch.relu(self.fc1(x))

        x = self.fc2(x)
        return x

# parameters
batch_size = 64
learning_rate = 0.001
num_epochs = 5

# Load the Fashion-MNIST dataset
transform = transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Create the CNN2D model and define the loss function and optimizer
model = CNN2D()
criterion = nn.CrossEntropyLoss()
#optimizer = optim.Adadelta(model.parameters(), lr=learning_rate)
#optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=True)
#optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=False)
#optimizer = optim.NAdam(model.parameters(), lr=learning_rate)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#optimizer = optim.Adagrad(model.parameters(), lr=learning_rate)
#optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
#optimizer = optim.RAdam(model.parameters(), lr=learning_rate)


# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item()))

# Test the mode
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy: {} %'.format(100 * correct / total))

