import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os

# 定义RNN模型
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(RNNModel, self).__init__()
        self.hidden_size = hidden_size
        
        # RNN层
        self.rnn = nn.Sequential(
            nn.Linear(input_size + hidden_size, hidden_size),
            nn.Tanh()
        )
        self.fc = nn.Linear(hidden_size, num_classes)
    
    def forward(self, x):
        # 初始化隐藏状态
        batch_size = x.size(0)
        hidden = torch.zeros(batch_size, self.hidden_size).to(x.device)
        
        # 序列处理
        seq_len = x.size(1)
        for t in range(seq_len):
            input_step = x[:, t, :]
            combined = torch.cat((input_step, hidden), dim=1)
            hidden = self.rnn(combined)
        
        # 最终输出
        out = self.fc(hidden)
        return out

# 训练函数
def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    total_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # 重塑输入: [batch, 784] -> [batch, 28, 28] (序列长度=28, 特征维度=28)
        data = data.view(-1, 28, 28)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}'
                  f' ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss

# 测试函数
def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(-1, 28, 28)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)}'
          f' ({accuracy:.0f}%)\n')
    return test_loss, accuracy

# 可视化预测结果
def visualize_predictions(model, device, test_loader, save_path='mnist_predictions.png'):
    model.eval()
    # 获取一个批次的数据（64张图片）
    data, target = next(iter(test_loader))
    data, target = data.to(device), target.to(device)
    
    # 选择前16张图片
    images = data[:16]
    true_labels = target[:16].cpu().numpy()
    
    # 准备模型输入
    input_data = images.view(-1, 28, 28)
    
    # 预测
    with torch.no_grad():
        output = model(input_data)
        pred_labels = output.argmax(dim=1)[:16].cpu().numpy()
    
    # 反归一化用于显示
    mean = 0.1307
    std = 0.3081
    images = images.cpu().numpy()
    images = images * std + mean  # 反归一化
    images = np.clip(images, 0, 1)  # 确保像素值在[0,1]范围内
    
    # 创建4x4的子图
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    fig.suptitle('MNIST Predictions', fontsize=16)
    
    for i, ax in enumerate(axes.flat):
        # 显示图像
        ax.imshow(images[i][0], cmap='gray')
        
        # 设置标题：绿色表示正确预测，红色表示错误预测
        color = 'green' if pred_labels[i] == true_labels[i] else 'red'
        ax.set_title(f"Pred: {pred_labels[i]}\nTrue: {true_labels[i]}", color=color, fontsize=12)
        ax.axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为总标题留出空间
    plt.savefig(save_path)
    print(f"Predictions saved to {save_path}")
    plt.show()

def main():
    # 超参数设置
    input_size = 28
    hidden_size = 128
    num_classes = 10
    batch_size = 64
    epochs = 10
    learning_rate = 0.001
    
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 创建保存模型的目录
    os.makedirs('models', exist_ok=True)
    
    # 数据加载
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(root='./data', 
                                   train=True, 
                                   transform=transform,
                                   download=True)
    
    test_dataset = datasets.MNIST(root='./data', 
                                  train=False, 
                                  transform=transform)
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                              batch_size=batch_size,
                                              shuffle=True)
    
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                             batch_size=batch_size,
                                             shuffle=False)
    
    # 初始化模型
    model = RNNModel(input_size, hidden_size, num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # 训练跟踪
    train_losses = []
    test_losses = []
    accuracies = []
    
    # 训练循环
    for epoch in range(1, epochs + 1):
        train_loss = train(model, device, train_loader, optimizer, criterion, epoch)
        test_loss, accuracy = test(model, device, test_loader, criterion)
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        accuracies.append(accuracy)
    
    # 保存训练好的模型
    model_path = 'models/mnist_rnn_model.pth'
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # 绘制损失曲线
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs+1), train_losses, 'b-', label='Training Loss')
    plt.plot(range(1, epochs+1), test_losses, 'r-', label='Test Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs+1), accuracies, 'g-')
    plt.title('Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('mnist_rnn_results.png')
    plt.show()
    
    # 可视化预测结果
    visualize_predictions(model, device, test_loader)

if __name__ == '__main__':
    main()