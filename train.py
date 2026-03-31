import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms ## check if indentation error is there in optimizer step.
import argparse
from model import ConvNeuralNet
import numpy as np 

import random
from torchvision.transforms import v2
from torch.optim.lr_scheduler import CosineAnnealingLR

def main(data_path, epochs, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

    def set_seed(seed_value):
        torch.manual_seed(seed_value)
        np.random.seed(seed_value)
        random.seed(seed_value)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed_value)
            torch.cuda.manual_seed_all(seed_value)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    SEED = 42
    set_seed(SEED)
    print(f"Manual seed set to {SEED}")
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
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
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
    unlabeled_set = torchvision.datasets.STL10(root='./data', split='unlabeled', download=True, transform=train_transform)
    unlabeled_loader = torch.utils.data.DataLoader(unlabeled_set, batch_size=160, shuffle=True)
    train_set = torchvision.datasets.STL10(root='./data', split= 'train', download=True, transform= transform)
    train_set_aug= torchvision.datasets.STL10(root='./data', split= 'train', download=True, transform=train_transform)
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
    train_loader_aug = torch.utils.data.DataLoader(train_set_aug, batch_size=32, shuffle=True)
    net=ConvNeuralNet()
    net.to(device)
    loss_function = nn.CrossEntropyLoss(label_smoothing= 0.1)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    scheduler = CosineAnnealingLR(optimizer, T_max=100)

    cutmix = v2.CutMix(num_classes=10)
    mixup = v2.MixUp(num_classes=10)
    epochs_plain = 20
    epochs_aug= 20
    epochs_cutmix= 40
    best_val_loss = float('inf')
    patience = 10
    counter = 0
    for epoch in range(epochs_plain):
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            running_loss += loss.item()
            if i % 50 == 1:
                print(f'[{epoch + 1}/{epochs_plain}, {i + 1:5d}] loss: {running_loss:.3f}')
                running_loss = 0.0

        optimizer.step()
    scheduler.step()

    for epoch in range(epochs_aug):
        running_loss = 0.0
        for i, data in enumerate(train_loader_aug):
            inputs, labels = data[0].to(device), data[1].to(device)
    
            optimizer.zero_grad()
            
            outputs = net(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            running_loss += loss.item()
            if i % 50 == 1:
                print(f'[{epoch + 1}/{epochs_aug}, {i + 1:5d}] loss: {running_loss :.3f}')
                running_loss = 0.0
    
            optimizer.step()
        scheduler.step()
    
    for epoch in range(epochs_cutmix):
        running_loss = 0.0
        for i, data in enumerate(train_loader_aug):
            inputs, labels = data[0].to(device), data[1].to(device)
    
            optimizer.zero_grad()
            if np.random.rand() < 0.5:
                
                cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])
                inputs, labels = cutmix_or_mixup(inputs, labels)
            outputs = net(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            running_loss += loss.item()
            if i % 50 == 1:
                print(f'[{epoch + 1}/{epochs_cutmix}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0
    
            optimizer.step()
        scheduler.step() 
    
    
    threshold = 0.9 
    unlabeled_iter = iter(unlabeled_loader)
    
    epochs_semi= 40
    
    patience = 10
    counter = 0
    for epoch in range(epochs_semi):
        #total loss=0??
        net.train()
        for i, (l_inputs, l_labels) in enumerate(train_loader):
            l_inputs, l_labels = l_inputs.to(device), l_labels.to(device)
            
            optimizer.zero_grad()
    
            
            outputs = net(l_inputs)
            supervised_loss = loss_function(outputs, l_labels)
    
            try:
                u_inputs, _ = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(unlabeled_loader)
                u_inputs, _ = next(unlabeled_iter)
            
            u_inputs = u_inputs.to(device)
            
         
            with torch.no_grad():
                u_outputs = net(u_inputs)
                probs = torch.softmax(u_outputs, dim=1)
                max_probs, pseudo_labels = torch.max(probs, dim=1)
                mask = max_probs > threshold  
            
            if mask.any():
                
                u_outputs_final = net(u_inputs[mask])
                unlabeled_loss = loss_function(u_outputs_final, pseudo_labels[mask])
                
               
                total_loss = supervised_loss + (0.5 * unlabeled_loss)
            else:
                total_loss = supervised_loss
            if i % 200 == 1:
                print(f'[{epoch + 1}/{epochs_semi}, {i + 1:5d}] loss: {supervised_loss.item() :.4f}')
    
            total_loss.backward()
            optimizer.step()
        scheduler.step()
    
    print("training completed")
    torch.save(net.state_dict(), 'marvel.pth')
    print("Model saved to marvel.pth")
        
        
       
    
        
    
    
        
    
    
        
    
    if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--data', type=str, default='./data', help='Path to dataset')
        parser.add_argument('--epochs', type=int, default=50)
        parser.add_argument('--batch_size', type=int, default=64)
        args = parser.parse_args()
        main(args.data, args.epochs, args.batch_size)
