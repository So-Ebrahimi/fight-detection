import torch
from torch.utils.data import DataLoader
from model import FightModel
from dataset import FightDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = FightDataset("dataset")
loader = DataLoader(dataset, batch_size=2, shuffle=True)

model = FightModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = torch.nn.CrossEntropyLoss()
counter = 0
for epoch in range(10):
    model.train()
    total_loss = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        pred = model(x)
        loss = criterion(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        print(f"Batch {counter+1} | Loss: {loss.item():.4f}")
        counter += 1

    print(f"Epoch {epoch+1} | Loss: {total_loss:.4f}")

torch.save(model.state_dict(), "fight_model.pth")
