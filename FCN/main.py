import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# 自定义Flatten层
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

# 定义FCN模型
def create_model():
    return nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(32, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Dropout(0.25),
        
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Dropout(0.25),
        
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(128, 128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Dropout(0.25),
        
        nn.Conv2d(128, 256, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(256, 256, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        Flatten(),
        nn.Linear(256, 10)
    )

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(train_loader.dataset)

def test(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def plot_loss(losses, filename='loss_curve.png'):
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def visualize_predictions(model, test_loader, device, filename='predictions.png'):
    model.eval()
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    # 获取整个测试集
    test_images = []
    test_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            test_images.append(inputs)
            test_labels.append(labels)
    
    # 合并所有批次数据
    all_images = torch.cat(test_images, dim=0)
    all_labels = torch.cat(test_labels, dim=0)
    
    # 随机选择16个不同的索引
    indices = torch.randperm(len(all_images))[:16]
    
    plt.figure(figsize=(10, 10))
    for i, idx in enumerate(indices):
        img = all_images[idx].cpu()
        true_label = all_labels[idx].item()
        
        # 反归一化图像以正确显示
        img = img / 2 + 0.5  # 反归一化 [0,1]
        img = img.permute(1, 2, 0).numpy()
        
        # 获取模型预测
        with torch.no_grad():
            output = model(all_images[idx].unsqueeze(0).to(device))
            _, predicted = torch.max(output, 1)
            predicted_label = predicted.item()
        
        # 绘制子图
        plt.subplot(4, 4, i+1)
        plt.imshow(img)
        plt.title(f'P: {classes[predicted_label]}\nT: {classes[true_label]}', fontsize=9)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def main():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 配置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # 加载数据集
    train_set = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)
    test_set = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=100, shuffle=False)
    
    # 创建模型
    model = create_model().to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    
    # 训练模型
    num_epochs = 80
    train_losses = []
    best_acc = 0.0
    
    print("Starting training...")
    for epoch in range(num_epochs):
        avg_loss = train(model, train_loader, criterion, optimizer, device)
        train_losses.append(avg_loss)
        
        acc = test(model, test_loader, device)
        scheduler.step(acc)
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Saved new best model with accuracy: {acc:.4f}")
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}")
    
    # 绘制并保存损失曲线
    plot_loss(train_losses)
    
    # 加载最佳模型
    model.load_state_dict(torch.load('best_model.pth', map_location=device))
    final_acc = test(model, test_loader, device)
    print(f"\nFinal Model Accuracy: {final_acc:.4f}")
    
    # 可视化预测结果
    visualize_predictions(model, test_loader, device)

if __name__ == "__main__":
    main()