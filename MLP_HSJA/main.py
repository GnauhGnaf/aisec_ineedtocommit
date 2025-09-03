import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from PIL import Image, ImageDraw, ImageTk
import tkinter as tk
from tkinter import messagebox

def load_data(batch_size=64):
    """加载MNIST数据集"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_set = datasets.MNIST(
        root='./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST(
        root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

class MLP(nn.Module):
    """多层感知机模型"""
    def __init__(self):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )
    
    def forward(self, x):
        return self.layers(x)

def train(model, device, train_loader, optimizer, epoch):
    """训练模型并返回平均损失"""
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    total_batches = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_batches += 1
        
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}'
                  f' ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    
    return total_loss / total_batches

def test(model, device, test_loader):
    """测试模型性能"""
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss(reduction='sum')
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    
    print(f'\nTest set: Average loss: {test_loss:.4f}, '
          f'Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')
    return test_loss, accuracy

def plot_losses(train_losses, test_losses):
    """绘制训练和测试损失曲线"""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Test Losses')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curve.png')
    plt.close()
    print("Loss curve saved as 'loss_curve.png'")

class DigitRecognizer:
    """实时手写数字识别GUI应用"""
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.window = tk.Tk()
        self.window.title("手写数字识别")
        
        # 创建画布
        self.canvas = tk.Canvas(self.window, width=280, height=280, bg="black")
        self.canvas.pack(pady=20)
        
        # 创建识别按钮
        self.recognize_btn = tk.Button(
            self.window, text="识别", command=self.recognize_digit)
        self.recognize_btn.pack(side=tk.LEFT, padx=20)
        
        # 创建清除按钮
        self.clear_btn = tk.Button(
            self.window, text="清除", command=self.clear_canvas)
        self.clear_btn.pack(side=tk.RIGHT, padx=20)
        
        # 结果显示标签
        self.result_label = tk.Label(
            self.window, text="请绘制数字", font=("Arial", 24))
        self.result_label.pack(pady=20)
        
        # 绑定鼠标事件
        self.canvas.bind("<B1-Motion>", self.paint)
        self.image = Image.new("L", (280, 280), 0)
        self.draw = ImageDraw.Draw(self.image)
        
    def paint(self, event):
        """处理鼠标绘制事件"""
        x, y = event.x, event.y
        r = 15  # 画笔半径
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="white", outline="white")
        self.draw.ellipse([x-r, y-r, x+r, y+r], fill=255)
    
    def clear_canvas(self):
        """清除画布内容"""
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="请绘制数字")
    
    def recognize_digit(self):
        """识别绘制的手写数字"""
        # 将图像转换为MNIST格式
        img = self.image.resize((28, 28))  # 缩小到28x28
        img = np.array(img)
        img = img.astype(np.float32) / 255.0
        img = (img - 0.1307) / 0.3081  # MNIST相同的归一化
        
        # 转换为PyTorch张量
        img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # 模型预测
        self.model.eval()
        with torch.no_grad():
            output = self.model(img_tensor)
            pred = output.argmax(dim=1, keepdim=True).item()
        
        # 显示结果
        self.result_label.config(text=f"识别结果: {pred}")
    
    def run(self):
        """启动GUI应用"""
        self.window.mainloop()

def main():
    # 设置超参数
    batch_size = 64
    epochs = 10
    learning_rate = 0.01
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载数据
    train_loader, test_loader = load_data(batch_size)
    
    # 初始化模型
    model = MLP().to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    
    # 记录训练和测试损失
    train_losses = []
    test_losses = []
    
    # 训练和测试循环
    best_accuracy = 0.0
    for epoch in range(1, epochs + 1):
        train_loss = train(model, device, train_loader, optimizer, epoch)
        test_loss, accuracy = test(model, device, test_loader)
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), "mnist_mlp_best.pth")
    
    # 绘制损失曲线
    plot_losses(train_losses, test_losses)
    print(f"Best test accuracy: {best_accuracy:.2f}%")
    
    # 加载最佳模型
    model.load_state_dict(torch.load("mnist_mlp_best.pth"))
    
    # 启动实时手写识别应用
    print("启动实时手写数字识别应用...")
    app = DigitRecognizer(model, device)
    app.run()

if __name__ == '__main__':
    main()