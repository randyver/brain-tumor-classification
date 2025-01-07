from torch import nn, optim
from src.config import device, LEARNING_RATE, EPOCHS, PATIENCE
from src.train import train_model
from src.preprocessing import train_loader, val_loader
from src.model import model

model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Train model
train_model(model, train_loader, val_loader, criterion, optimizer, device, epochs=EPOCHS, patience=PATIENCE)