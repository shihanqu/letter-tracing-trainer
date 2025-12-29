#!/usr/bin/env python3
"""
EMNIST Balanced Model Training and ONNX Conversion

This script trains a CNN on EMNIST Balanced dataset and exports to ONNX format.
Optimized for M1 Mac with MPS acceleration.

Usage:
    python train_model.py
    
Requirements:
    pip install torch torchvision onnx
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import time

# Check for MPS (Metal Performance Shaders) on M1 Mac
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("✓ Using MPS (Metal) acceleration")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("✓ Using CUDA acceleration")
else:
    device = torch.device("cpu")
    print("⚠ Using CPU (slower)")

# EMNIST Balanced class mapping (47 classes)
# Classes 0-9: digits 0-9
# Classes 10-35: uppercase A-Z
# Classes 36-46: lowercase letters that differ from uppercase
EMNIST_BALANCED_LABELS = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'd', 'e',
    'f', 'g', 'h', 'n', 'q', 'r', 't'
]

class EMNISTNet(nn.Module):
    """CNN architecture for EMNIST classification."""
    
    def __init__(self, num_classes=47):
        super(EMNISTNet, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling and dropout
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
        # Activation
        self.relu = nn.ReLU()
        
    def forward(self, x):
        # Conv block 1: 28x28 -> 14x14
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        
        # Conv block 2: 14x14 -> 7x7
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout1(x)
        
        # Conv block 3: 7x7 -> 3x3
        x = self.relu(self.conv3(x))
        x = self.pool(x)
        x = self.dropout1(x)
        
        # Flatten and FC
        x = x.view(-1, 128 * 3 * 3)
        x = self.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        
        return x


def load_emnist_balanced():
    """Load EMNIST Balanced dataset with preprocessing."""
    
    # EMNIST images need to be transposed (they come rotated/flipped)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.transpose(1, 2).flip(2)),  # Fix orientation
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST-like normalization
    ])
    
    print("Downloading EMNIST Balanced dataset...")
    
    train_dataset = datasets.EMNIST(
        root='./data',
        split='balanced',
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = datasets.EMNIST(
        root='./data',
        split='balanced',
        train=False,
        download=True,
        transform=transform
    )
    
    print(f"✓ Training samples: {len(train_dataset)}")
    print(f"✓ Test samples: {len(test_dataset)}")
    
    return train_dataset, test_dataset


def train_model(model, train_loader, test_loader, epochs=15):
    """Train the model."""
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    best_accuracy = 0
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        start_time = time.time()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            if batch_idx % 100 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}", end='\r')
        
        train_acc = 100. * correct / total
        epoch_time = time.time() - start_time
        
        # Evaluate on test set
        test_acc = evaluate_model(model, test_loader)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Acc: {train_acc:.2f}% - Test Acc: {test_acc:.2f}% - Time: {epoch_time:.1f}s")
        
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            torch.save(model.state_dict(), 'models/emnist_balanced_best.pth')
            print(f"  ✓ New best model saved! ({test_acc:.2f}%)")
        
        scheduler.step()
    
    return best_accuracy


def evaluate_model(model, test_loader):
    """Evaluate model accuracy."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    return 100. * correct / total


def export_to_onnx(model, output_path='../models/emnist_balanced.onnx'):
    """Export model to ONNX format."""
    
    model.eval()
    model.to('cpu')  # ONNX export works best on CPU
    
    # Dummy input for tracing
    dummy_input = torch.randn(1, 1, 28, 28)
    
    print(f"\nExporting to ONNX: {output_path}")
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"✓ ONNX model saved to {output_path}")
    
    # Verify ONNX model
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model verification passed!")
    except ImportError:
        print("⚠ Install 'onnx' package to verify model: pip install onnx")


def save_label_mapping(output_path='../models/labels.json'):
    """Save label mapping for use in the webapp."""
    import json
    
    with open(output_path, 'w') as f:
        json.dump(EMNIST_BALANCED_LABELS, f)
    
    print(f"✓ Label mapping saved to {output_path}")


def main():
    print("=" * 50)
    print("EMNIST Balanced Model Training")
    print("=" * 50)
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Load data
    train_dataset, test_dataset = load_emnist_balanced()
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=0)
    
    # Create model
    model = EMNISTNet(num_classes=47).to(device)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train
    print("\n" + "=" * 50)
    print("Training...")
    print("=" * 50)
    
    best_accuracy = train_model(model, train_loader, test_loader, epochs=15)
    
    print(f"\n✓ Training complete! Best accuracy: {best_accuracy:.2f}%")
    
    # Load best model and export
    model.load_state_dict(torch.load('models/emnist_balanced_best.pth'))
    export_to_onnx(model)
    save_label_mapping()
    
    print("\n" + "=" * 50)
    print("Done! Files created:")
    print("  - models/emnist_balanced.onnx")
    print("  - models/labels.json")
    print("=" * 50)


if __name__ == '__main__':
    main()
