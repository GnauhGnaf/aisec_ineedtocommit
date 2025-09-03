import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# 定义YOLO风格卷积神经网络
class YOLO_CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(YOLO_CNN, self).__init__()
        self.features = nn.Sequential(
            # 卷积块1
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            
            # 卷积块2
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            
            # 卷积块3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            
            # 卷积块4
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2),
            
            # 卷积块5
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2, 2)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256 * 1 * 1, 512),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# 训练函数
def train_model(model, train_loader, criterion, optimizer, device):
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
    epoch_loss = running_loss / len(train_loader.dataset)
    return epoch_loss

# 测试函数
def test_model(model, test_loader, device):
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
    accuracy = 100 * correct / total
    return accuracy

# 可视化结果函数
def visualize_results(images, true_labels, pred_labels, class_names, save_path):
    plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.5, hspace=0.75)
    
    for i in range(16):
        ax = plt.subplot(gs[i])
        img = images[i].permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        ax.imshow(img)
        ax.axis('off')
        
        true_name = class_names[true_labels[i]]
        pred_name = class_names[pred_labels[i]]
        color = 'green' if true_labels[i] == pred_labels[i] else 'red'
        
        ax.set_title(f"True: {true_name}\nPred: {pred_name}", 
                     fontsize=8, color=color)
    
    plt.suptitle('CIFAR-10 Classification Results', fontsize=16)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

# 主函数
def main():
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 超参数设置
    batch_size = 128
    learning_rate = 0.001
    num_epochs = 80
    
    # 数据预处理
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # 加载CIFAR-10数据集
    train_set = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # 类别名称
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    # 初始化模型
    model = YOLO_CNN(num_classes=10).to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3, verbose=True)
    
    # 训练模型
    train_losses = []
    test_accuracies = []
    best_acc = 0.0
    
    print("Starting training...")
    for epoch in range(num_epochs):
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        test_acc = test_model(model, test_loader, device)
        
        train_losses.append(train_loss)
        test_accuracies.append(test_acc)
        scheduler.step(test_acc)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] - "
              f"Train Loss: {train_loss:.4f}, Test Acc: {test_acc:.2f}%")
        
        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'best_model.pth')
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curve.png')
    plt.close()
    
    # 加载最佳模型
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    
    # 随机选择16张测试图片
    indices = np.random.choice(len(test_set), 16, replace=False)
    test_subset = torch.utils.data.Subset(test_set, indices)
    test_loader_sub = torch.utils.data.DataLoader(
        test_subset, batch_size=16, shuffle=False)
    
    images, labels = next(iter(test_loader_sub))
    images = images.to(device)
    
    # 预测
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
    
    # 反归一化图像
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(3, 1, 1)
    images_denorm = images.cpu() * std + mean
    images_denorm = torch.clamp(images_denorm, 0, 1)
    
    # 可视化结果
    visualize_results(images_denorm, labels.cpu().numpy(), 
                     predicted.cpu().numpy(), class_names, 
                     'classification_results.png')
    
    print("Training completed. Results saved to:")
    print("- loss_curve.png: Training loss curve")
    print("- best_model.pth: Best model weights")
    print("- classification_results.png: 4x4 classification results")

if __name__ == "__main__":
    main()