import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import random
from torchvision.transforms import v2
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.ao.quantization
import argparse


# Set the quantization engine
if 'fbgemm' in torch.backends.quantized.supported_engines:
    torch.backends.quantized.engine = 'fbgemm'
elif 'qnnpack' in torch.backends.quantized.supported_engines:
    torch.backends.quantized.engine = 'qnnpack'
else:
    print("No supported quantization engine found!")
# --- 1. SETUP & SEEDING ---

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.backends.mps.is_built() and torch.backends.mps.is_available():
    device = torch.device("mps")

def set_seed(seed_value=42):
    torch.manual_seed(seed_value)
    np.random.seed(seed_value)
    random.seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)

# --- 2. DATA PREPARATION ---
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(96, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Datasets
unlabeled_set = torchvision.datasets.STL10(root='./data', split='unlabeled', download=True, transform=train_transform)
train_set = torchvision.datasets.STL10(root='./data', split='train', download=True, transform=test_transform) # Standard
train_set_aug = torchvision.datasets.STL10(root='./data', split='train', download=True, transform=train_transform) # Augment
test_set = torchvision.datasets.STL10(root='./data', split='test', download=True, transform=test_transform)

# Loaders
unlabeled_loader = torch.utils.data.DataLoader(unlabeled_set, batch_size=192, shuffle=True)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
train_loader_aug = torch.utils.data.DataLoader(train_set_aug, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)

# --- 3. MODEL DEFINITION ---
class ConvNeuralNet(nn.Module):
    def __init__(self): # Fixed double underscores
        super().__init__()
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), # 0, 1, 2
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), # 3, 4, 5
            nn.MaxPool2d(2), # 6
            
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), # 7, 8, 9
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), # 10, 11, 12
            nn.MaxPool2d(2), # 13
            
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), # 14, 15, 16
            nn.MaxPool2d(2), # 17
        )
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.quant(x)
        x = self.features(x)
        x = self.gap(x)
        x = self.classifier(x)
        x = self.dequant(x)
        return x

net = ConvNeuralNet().to(device)

# --- 4. TRAINING HYPERPARAMETERS ---
loss_function = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.Adam(net.parameters(), lr=0.001)
scheduler = CosineAnnealingLR(optimizer, T_max=100)
cutmix = v2.CutMix(num_classes=10)
mixup = v2.MixUp(num_classes=10)

# --- 5. TRAINING LOOPS (Simplified) ---
def train_epoch(loader, use_mix=False):
    net.train()
    for i, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        if use_mix and np.random.rand() < 0.5:
            mix_op = v2.RandomChoice([cutmix, mixup])
            inputs, labels = mix_op(inputs, labels)
            
        outputs = net(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

print("Starting Training Phases...")
# Phase 1: Plain
for epoch in range(20): train_epoch(train_loader); scheduler.step();print('step 1 done')
# Phase 2: Augmented
for epoch in range(20): train_epoch(train_loader_aug); scheduler.step();print('step 2 done')
# Phase 3: Cutmix/Mixup
for epoch in range(20): train_epoch(train_loader_aug, use_mix=True); scheduler.step();print('step 3 done')

# --- 6. SEMI-SUPERVISED LEARNING PHASE ---
unlabeled_iter = iter(unlabeled_loader)
threshold = 0.9
print('Semisupervised started')
for epoch in range(40):
    net.train()
    for i, (l_inputs, l_labels) in enumerate(train_loader):
        l_inputs, l_labels = l_inputs.to(device), l_labels.to(device)
        
        # Supervised
        optimizer.zero_grad()
        sup_outputs = net(l_inputs)
        sup_loss = loss_function(sup_outputs, l_labels)
        
        # Pseudo-labeling
        try: u_inputs, _ = next(unlabeled_iter)
        except StopIteration: 
            unlabeled_iter = iter(unlabeled_loader)
            u_inputs, _ = next(unlabeled_iter)
        
        u_inputs = u_inputs.to(device)
        with torch.no_grad():
            u_logits = net(u_inputs)
            probs = torch.softmax(u_logits, dim=1)
            max_probs, pseudo_labels = torch.max(probs, dim=1)
            mask = max_probs > threshold
            
        if mask.any():
            u_outputs = net(u_inputs[mask])
            unsup_loss = loss_function(u_outputs, pseudo_labels[mask])
            total_loss = sup_loss + (0.5 * unsup_loss)
        else:
            total_loss = sup_loss
            
        total_loss.backward()
        optimizer.step()
    scheduler.step()
    print('epoch completed')

# Save unquantized model
torch.save(net.state_dict(), 'marvel_fp32.pth')

# --- 7. QUANTIZATION (PTQ) ---
print("\n--- Starting Quantization ---")
net.to('cpu')
net.eval()

# Fusion (Triplets: Conv, BN, ReLU)
torch.ao.quantization.fuse_modules(net.features, ['0', '1', '2'], inplace=True)
torch.ao.quantization.fuse_modules(net.features, ['3', '4', '5'], inplace=True)
torch.ao.quantization.fuse_modules(net.features, ['7', '8', '9'], inplace=True)
torch.ao.quantization.fuse_modules(net.features, ['10', '11', '12'], inplace=True)
torch.ao.quantization.fuse_modules(net.features, ['14', '15', '16'], inplace=True)

# Config
engine = 'fbgemm' if 'fbgemm' in torch.backends.quantized.supported_engines else 'qnnpack'
net.qconfig = torch.ao.quantization.get_default_qconfig(engine)

# Calibration
net_prepared = torch.ao.quantization.prepare(net)
with torch.no_grad():
    for i, (imgs, _) in enumerate(train_loader):
        net_prepared(imgs)
        if i >= 10: break

net_quantized = torch.ao.quantization.convert(net_prepared)
torch.save(net_quantized.state_dict(), 'marvel_int8.pth')

# --- 8. ACCURACY EVALUATION ---
def evaluate(model, loader, name="Model"):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            # Note: Quantized model MUST stay on CPU
            outputs = model(imgs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = 100 * correct / total
    print(f'Accuracy of {name}: {acc:.2f}%')
    return acc

# Evaluation (both on CPU for fair comparison)
evaluate(net, test_loader, "FP32 Model (Fused)")
evaluate(net_quantized, test_loader, "INT8 Quantized Model")

# Size Comparison
size_fp32 = os.path.getsize('marvel_fp32.pth') / 1e6
size_int8 = os.path.getsize('marvel_int8.pth') / 1e6
print(f"\nSize Comparison:\nFP32: {size_fp32:.2f} MB\nINT8: {size_int8:.2f} MB")
        
        
       
    
        
    
    
        
    
    
        
    
    if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--data', type=str, default='./data', help='Path to dataset')
        parser.add_argument('--epochs', type=int, default=50)
        parser.add_argument('--batch_size', type=int, default=64)
        args = parser.parse_args()
        main(args.data, args.epochs, args.batch_size)
