import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.metrics import classification_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.Dropout(0.5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(32*14*14, 10)
        )

    def forward(self, x):
        return self.model(x)

transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
validation_dataset, test_dataset = random_split(test_dataset, [0.5, 0.5])
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=16, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

best_model = None
best_val_loss = float('inf')
num_epochs = 10
for epoch in range(num_epochs):
    train_loss = 0
    model.train()
    for X, y in train_dataloader:
        optimizer.zero_grad()
        X = X.to(device)
        y = y.to(device)
        y_pred = model(X)
        loss = criterion(y_pred, y)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

    train_loss /= len(train_dataloader)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X, y in validation_dataloader:
            X = X.to(device)
            y = y.to(device)
            y_pred = model(X)
            loss = criterion(y_pred, y)
            val_loss += loss.item()
    val_loss /= len(validation_dataloader)

    if best_val_loss > val_loss:
        best_model = model.state_dict()
        best_val_loss = val_loss

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

torch.save(best_model, 'best_model.pt')
best_model = CNN().to(device)
best_model.load_state_dict(torch.load('best_model.pt'))

y_true = []
y_pred = []
best_model.eval()
with torch.no_grad():
    for X, y in test_dataloader:
        y_true.extend(y.cpu().numpy())
        X = X.to(device)
        y_pred.extend(best_model(X).argmax(dim=1).cpu().numpy())

print(classification_report(y_true, y_pred))