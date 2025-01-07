from tqdm import tqdm
import torch
import matplotlib.pyplot as plt

def train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=10, patience=3):
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    early_stop = False

    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]", leave=False)
        
        for images, labels in progress_bar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
        
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        val_accuracy = 100 * correct / total
        val_accuracies.append(val_accuracy)

        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), '../model/best_model.pth')
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print("Early stopping triggered. Stopping training.")
            early_stop = True
            break

    if early_stop:
        model.load_state_dict(torch.load('../model/best_model.pth'))
        print("Loaded best model based on validation loss.")

    plot_training_history(train_losses, val_losses, val_accuracies)
    
    return model

def plot_training_history(train_losses, val_losses, val_accuracies):
    epochs = len(train_losses)
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot Loss
    axs[0].plot(range(1, epochs+1), train_losses, label='Train Loss', color='blue')
    axs[0].plot(range(1, epochs+1), val_losses, label='Val Loss', color='red')
    axs[0].set_title('Loss over Epochs')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    
    # Plot Accuracy
    axs[1].plot(range(1, epochs+1), val_accuracies, label='Val Accuracy', color='green')
    axs[1].set_title('Validation Accuracy over Epochs')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Accuracy (%)')
    axs[1].legend()
    
    plt.tight_layout()
    plt.show()