import torch
from torch.utils.data import DataLoader
from model import FightModel
from dataset import FightDataset

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load train and validation datasets
print("Loading datasets...")
train_dataset = FightDataset("dataset/train")
valid_dataset = FightDataset("dataset/valid")

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=8, shuffle=False)

print(f"Train samples: {len(train_dataset)}")
print(f"Validation samples: {len(valid_dataset)}")

# Create model
model = FightModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()

# Training loop
num_epochs = 30
best_val_loss = float('inf')

print("\nStarting training...")
print("=" * 60)

counter = 0
for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        pred = model(x)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(pred.data, 1)
        train_total += y.size(0)
        train_correct += (predicted == y).sum().item()
        print(f"Batch {counter+1} | Loss: {loss.item():.4f}")
        counter += 1
    train_acc = 100 * train_correct / train_total
    avg_train_loss = train_loss / len(train_loader)

    # Validation phase
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x)
            loss = criterion(pred, y)

            val_loss += loss.item()
            _, predicted = torch.max(pred.data, 1)
            val_total += y.size(0)
            val_correct += (predicted == y).sum().item()

    val_acc = 100 * val_correct / val_total
    avg_val_loss = val_loss / len(valid_loader)

    # Print epoch results
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"  Valid Loss: {avg_val_loss:.4f} | Valid Acc: {val_acc:.2f}%")

    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "fight_model.pth")
        print(f"  ✓ Saved best model (val_loss: {avg_val_loss:.4f})")
    
    print("-" * 60)

print("\nTraining complete!")
print(f"Best validation loss: {best_val_loss:.4f}")
print("Model saved to: fight_model.pth")
