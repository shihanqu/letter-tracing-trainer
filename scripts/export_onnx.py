#!/usr/bin/env python3
"""
Convert trained PyTorch model to ONNX format.
"""

import torch
import torch.nn as nn
import os

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

def main():
    print("Loading trained model...")
    
    # Load the trained model
    model = EMNISTNet(num_classes=47)
    model.load_state_dict(torch.load('models/emnist_balanced_best.pth', weights_only=True))
    model.eval()
    model.cpu()
    
    print("Exporting to ONNX...")
    
    # Dummy input for tracing
    dummy_input = torch.randn(1, 1, 28, 28)
    
    output_path = '../models/emnist_balanced.onnx'
    
    # Export with legacy method (dynamo=False) for better compatibility
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
        },
        dynamo=False  # Use legacy export
    )
    
    print(f"✓ ONNX model saved to {output_path}")
    
    # Get file size
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"✓ Model size: {size_mb:.2f} MB")
    
    # Verify ONNX model
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model verification passed!")
    except Exception as e:
        print(f"⚠ ONNX verification warning: {e}")
    
    print("\nDone! The model is ready for use in the webapp.")

if __name__ == '__main__':
    main()
