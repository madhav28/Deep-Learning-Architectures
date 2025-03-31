import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
from torchvision import datasets, transforms
from sklearn.metrics import classification_report

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FNN(nn.Module):
    def __init__(self, input_size=784, hidden_size=64, output_size=10, dropout_rate=0.5):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(hidden_size, output_size),
        )

    def forward(self, X):
        return self.model(X)

transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])
train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)
validation_dataset, test_dataset = random_split(test_dataset, [0.5, 0.5])

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=32, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = FNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

best_model = None
best_val_loss = float('inf')
best_epoch = 0
num_epochs = 20
for epoch in range(num_epochs):
    running_loss = 0
    for X, y in train_dataloader:
        optimizer.zero_grad()
        X = X.to(device)
        y = y.to(device)
        y_pred = model(X)
        loss = criterion(y_pred, y)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
    
    running_loss /= len(train_dataloader)
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
        best_val_loss = val_loss
        best_model = model.state_dict()
        best_epoch = epoch+1

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {running_loss:.4f}, Validation Loss: {val_loss: .4f}")

print(f"Best epoch is {best_epoch}")


torch.save(best_model, 'best_model.pt')
best_model = FNN().to(device)
best_model.load_state_dict(torch.load('best_model.pt'))
best_model.eval()

y_pred = []
y_true = []
with torch.no_grad():
    for X, y in test_dataloader:
        X = X.to(device)
        y_true.extend(y.cpu().tolist())
        y_pred.extend(best_model(X).argmax(dim=1).cpu().tolist())

print(classification_report(y_true, y_pred))