import torch
import torchvision
import torchvision.transforms as transforms
import argparse
from model import ConvNeuralNet

def evaluate(data_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_set = torchvision.datasets.STL10(root=data_path, split='test', download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)

    net = ConvNeuralNet().to(device)
    net.load_state_dict(torch.load("model.pth", map_location=device))
    net.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy on test images: {100 * correct / total:.2f}%')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='./data')
    args = parser.parse_args()
    evaluate(args.data)
