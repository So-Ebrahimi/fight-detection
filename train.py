import torch
from torch.utils.data import DataLoader
from model import FightModel
from dataset import FightDataset
import os
import sys

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load train and validation datasets with augmentation for training
    print("Loading datasets...")
    train_dataset = FightDataset("dataset/train", augment=True, img_size=224)
    valid_dataset = FightDataset("dataset/valid", augment=False, img_size=224)

    # Set num_workers to 0 on Windows to avoid multiprocessing issues
    # On Linux/Mac, you can use num_workers > 0 for faster loading
    num_workers = 0 if sys.platform == 'win32' else 2
    
    # Increase batch size if GPU memory allows, otherwise keep it smaller
    batch_size = 4 if device == "cuda" else 2
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=True if device == "cuda" else False)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, 
                             num_workers=num_workers, pin_memory=True if device == "cuda" else False)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(valid_dataset)}")
    print(f"Batch size: {batch_size}")

    # Create model
    model = FightModel().to(device)

    # Better optimizer with weight decay for regularization
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Learning rate scheduler - reduce on plateau (removed verbose parameter)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )

    # Loss function
    criterion = torch.nn.CrossEntropyLoss()

    # Training configuration
    num_epochs = 50
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience = 10
    patience_counter = 0
    gradient_clip = 1.0  # Gradient clipping to prevent exploding gradients

    print("\nStarting training...")
    print("=" * 60)

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            pred = model(x)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(pred.data, 1)
            train_total += y.size(0)
            train_correct += (predicted == y).sum().item()
            
            # Print progress less frequently
            if (batch_idx + 1) % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1}/{num_epochs} | Batch {batch_idx+1}/{len(train_loader)} | "
                      f"Loss: {loss.item():.4f} | LR: {current_lr:.6f}")
        
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
        
        # Update learning rate
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(avg_val_loss)
        new_lr = optimizer.param_groups[0]['lr']
        
        # Print epoch results
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Valid Loss: {avg_val_loss:.4f} | Valid Acc: {val_acc:.2f}%")
        print(f"  Learning Rate: {new_lr:.6f}", end="")
        if new_lr < old_lr:
            print(f" (reduced from {old_lr:.6f})")
        else:
            print()

        # Save best model based on validation accuracy (better metric than loss)
        if val_acc > best_val_acc or (val_acc == best_val_acc and avg_val_loss < best_val_loss):
            if val_acc > best_val_acc:
                best_val_acc = val_acc
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "fight_model.pth")
            print(f"  ✓ Saved best model (val_acc: {val_acc:.2f}%, val_loss: {avg_val_loss:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{patience})")
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            print(f"Best validation accuracy: {best_val_acc:.2f}%")
            break
        
        print("-" * 60)

    print("\nTraining complete!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("Model saved to: fight_model.pth")

if __name__ == '__main__':
    main()
