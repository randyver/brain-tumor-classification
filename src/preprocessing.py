from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from .config import BATCH_SIZE
from torchvision import datasets
import os

# Data augmentation
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(), 
])

# Load datasets
base_dir = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(base_dir, "../dataset/Training")
test_path = os.path.join(base_dir, "../dataset/Testing")

train_dataset = datasets.ImageFolder(train_path, transform=transform)
test_dataset = datasets.ImageFolder(test_path, transform=transform)

# Split training data into train and validation sets
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# Get class labels
classes = train_dataset.dataset.classes
print("Classes:", classes)

# Print dataset sizes
print(f"Training Samples: {len(train_dataset)}, Validation Samples: {len(val_dataset)}, Testing Samples: {len(test_dataset)}")