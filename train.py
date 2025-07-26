
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from PIL import Image
import numpy as np

# Configuration
class Config:
    DEFORESTATION_TRAIN_DIR = 'dataset/deforestation/train'
    DEFORESTATION_VAL_DIR = 'dataset/deforestation/val'
    WILDFIRE_TRAIN_DIR = 'dataset/wildfire/train'
    WILDFIRE_VAL_DIR = 'dataset/wildfire/val'
    IMG_SIZE = 224
    BATCH_SIZE = 32
    EPOCHS = 15
    LEARNING_RATE = 0.0001
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FloatImageFolder(ImageFolder):
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, torch.tensor(target, dtype=torch.float32)

class WildfireDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['no_fire', 'fire']
        self.samples = []
        
        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                self.samples.append((img_path, class_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        return img, torch.tensor(label, dtype=torch.float32)

def get_transforms():
    train_transform = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(Config.IMG_SIZE, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def create_deforestation_model():
    model = models.mobilenet_v2(pretrained=True)
    for param in model.features.parameters():
        param.requires_grad = False
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.last_channel, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
        nn.Sigmoid()
    )
    return model.to(Config.DEVICE)

def create_wildfire_model():
    model = models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Sequential(
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 1),
        nn.Sigmoid()
    )
    return model.to(Config.DEVICE)

def train_model(model, train_loader, val_loader, model_name):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    best_val_acc = 0.0
    
    for epoch in range(Config.EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(Config.DEVICE)
            labels = labels.to(Config.DEVICE).unsqueeze(1)  # Ensure shape [batch_size, 1]
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        train_acc = 100 * correct / total
        train_loss = running_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(Config.DEVICE)
                labels = labels.to(Config.DEVICE).unsqueeze(1)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                preds = (outputs > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        val_acc = 100 * val_correct / val_total
        val_loss = val_loss / len(val_loader)
        
        print(f"{model_name} - Epoch {epoch+1}/{Config.EPOCHS} => "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% || "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{model_name}_model.pth")
            print(f"✅ Saved new best {model_name} model with val acc: {val_acc:.2f}%")

def main():
    # Check directories
    for dir_path in [Config.DEFORESTATION_TRAIN_DIR, Config.DEFORESTATION_VAL_DIR,
                    Config.WILDFIRE_TRAIN_DIR, Config.WILDFIRE_VAL_DIR]:
        if not os.path.exists(dir_path):
            raise FileNotFoundError(f"Directory not found: {dir_path}")
    
    # Get transforms
    train_transform, val_transform = get_transforms()
    
    # Train deforestation model
    print("\nTraining Deforestation Model...")
    deforestation_train = FloatImageFolder(Config.DEFORESTATION_TRAIN_DIR, transform=train_transform)
    deforestation_val = FloatImageFolder(Config.DEFORESTATION_VAL_DIR, transform=val_transform)
    
    deforestation_train_loader = DataLoader(deforestation_train, batch_size=Config.BATCH_SIZE, shuffle=True)
    deforestation_val_loader = DataLoader(deforestation_val, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    deforestation_model = create_deforestation_model()
    train_model(deforestation_model, deforestation_train_loader, deforestation_val_loader, "deforestation")
    
    # Train wildfire model
    print("\nTraining Wildfire Model...")
    wildfire_train = WildfireDataset(Config.WILDFIRE_TRAIN_DIR, transform=train_transform)
    wildfire_val = WildfireDataset(Config.WILDFIRE_VAL_DIR, transform=val_transform)
    
    wildfire_train_loader = DataLoader(wildfire_train, batch_size=Config.BATCH_SIZE, shuffle=True)
    wildfire_val_loader = DataLoader(wildfire_val, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    wildfire_model = create_wildfire_model()
    train_model(wildfire_model, wildfire_train_loader, wildfire_val_loader, "wildfire")

if __name__ == '__main__':
    main()