import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from main import MLP  # 从训练文件中导入模型定义

def load_test_data():
    """加载MNIST测试集"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_set = datasets.MNIST(
        root='./data', train=False, download=True, transform=transform)
    return test_set

def denormalize(tensor):
    """反归一化图像以便显示"""
    tensor = tensor.clone()  # 避免修改原始张量
    mean = torch.tensor([0.1307])
    std = torch.tensor([0.3081])
    tensor = tensor * std.view(1, 1, 1) + mean.view(1, 1, 1)
    tensor = tensor.clamp(0, 1)  # 确保值在0-1范围内
    return tensor

def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    model_path = "mnist_mlp_best.pth"
    model = MLP().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"模型已从 '{model_path}' 加载")
    
    # 加载测试数据
    test_set = load_test_data()
    print(f"测试集大小: {len(test_set)}")
    
    # 随机选择16个样本
    indices = np.random.choice(len(test_set), 16, replace=False)
    images = []
    labels = []
    
    # 收集图像和标签
    for idx in indices:
        image, label = test_set[idx]
        images.append(image)
        labels.append(label)
    
    # 转换为批量张量
    images = torch.stack(images).to(device)
    
    # 进行预测
    with torch.no_grad():
        outputs = model(images)
        _, predictions = torch.max(outputs, 1)
    
    # 转换为CPU numpy数组
    images = images.cpu()
    predictions = predictions.cpu().numpy()
    
    # 创建4x4子图
    plt.figure(figsize=(12, 12))
    plt.suptitle("MNIST prediction", fontsize=20)
    
    for i in range(16):
        plt.subplot(4, 4, i+1)
        
        # 反归一化并显示图像
        img = denormalize(images[i]).squeeze(0).numpy()
        plt.imshow(img, cmap='gray')
        
        # 设置标题（预测和真实标签）
        pred = predictions[i]
        true_label = labels[i]
        
        # 根据预测是否正确设置标题颜色
        title_color = 'green' if pred == true_label else 'red'
        plt.title(f"pred: {pred}\ntrue: {true_label}", color=title_color)

        plt.axis('off')
    
    # 调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为总标题留出空间
    
    # 保存图像
    plt.savefig("mnist_predictions.png", dpi=150, bbox_inches='tight')
    print("预测结果已保存为 'mnist_predictions.png'")
    
    # 显示图像
    plt.show()

if __name__ == '__main__':
    main()