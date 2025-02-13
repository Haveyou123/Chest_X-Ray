import numpy as np
from Dataset import *
import torch
import torch.nn as nn
from torchvision import transforms, models
import torch.optim as optim
from sklearn.metrics import accuracy_score
device = ("cuda" if torch.cuda.is_available() else "cpu")


class AdvancedCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.conv1 = self.make_block(3, 16)
        self.conv2 = self.make_block(16, 32)
        self.conv3 = self.make_block(32, 64)
        self.conv4 = self.make_block(64, 128)
        self.conv5 = self.make_block(128, 256)
        
        self.fc1 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=7*7*256, out_features=2048),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=2048, out_features=1024),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(in_features=1024, out_features=2)
        )

    def make_block(self, in_channel, out_channel, kernel_size = 3, stride = 1, padding = 1):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(num_features=out_channel),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(num_features=out_channel),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

model = AdvancedCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 10

for epoch in range(num_epochs):
    running_loss = 0.0  # Reset loss for each epoch

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()  # Convert tensor to scalar

    print(f"Epoch: {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}")

    model.eval()
    val_labels = []
    val_preds = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            output = model(images)
            _, preds = torch.max(output, 1)
            val_labels.extend(labels.cpu().numpy())
            val_preds.extend(preds.cpu().numpy())

    val_accuracy = accuracy_score(val_labels, val_preds)
    print("Validation accuracy: ", val_accuracy)

model.eval()
test_labels = []
test_preds = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        output = model(images)
        _, preds = torch.max(output, 1)
        test_labels.extend(labels.cpu().numpy())
        test_preds.extend(preds.cpu().numpy())

test_accuracy = accuracy_score(test_labels, test_preds)
print("test accuracy: ", test_accuracy)

torch.save(model.state_dict(), "pneumonia_classifier.pth")