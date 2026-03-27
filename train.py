import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
from model import ConvNeuralNet

def main(data_path, epochs, batch_size):
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop(96, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Datasets
    train_set = torchvision.datasets.STL10(root=data_path, split='train', download=True, transform=train_transform)
    unlabeled_set = torchvision.datasets.STL10(root=data_path, split='unlabeled', download=True, transform=train_transform)
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    unlabeled_loader = torch.utils.data.DataLoader(unlabeled_set, batch_size=batch_size*2, shuffle=True, num_workers=2)

    net = ConvNeuralNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    
    # Semi-supervised setup
    threshold = 0.95
    
    for epoch in range(epochs):
        net.train()
        u_iter = iter(unlabeled_loader)
        
        for i, (l_inputs, l_labels) in enumerate(train_loader):
            l_inputs, l_labels = l_inputs.to(device), l_labels.to(device)
            optimizer.zero_grad()

            # 1. Supervised Loss
            outputs = net(l_inputs)
            loss = criterion(outputs, l_labels)

            # 2. Pseudo-labeling (SSL)
            try:
                u_inputs, _ = next(u_iter)
                u_inputs = u_inputs.to(device)
                with torch.no_grad():
                    u_outputs = net(u_inputs)
                    probs = torch.softmax(u_outputs, dim=1)
                    max_probs, pseudo_labels = torch.max(probs, dim=1)
                    mask = max_probs > threshold
                
                if mask.any():
                    u_loss = criterion(net(u_inputs[mask]), pseudo_labels[mask])
                    loss += 0.5 * u_loss
            except StopIteration:
                u_iter = iter(unlabeled_loader)

            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}/{epochs} completed.")

    torch.save(net.state_dict(), "model.pth")
    print("Model saved to model.pth")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./data', help='Path to dataset')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()
    main(args.data, args.epochs, args.batch_size)
