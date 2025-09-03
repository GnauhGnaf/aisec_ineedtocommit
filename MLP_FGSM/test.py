import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 加载MNIST测试数据
def load_mnist_test(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_set = datasets.MNIST(
        root='./data', train=False, download=True, transform=transform)
    
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False)
    
    return test_loader

# 定义MLP模型
class MLP(nn.Module):
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

# 评估模型在原始测试集上的性能
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    accuracy = 100. * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy

# FGSM攻击实现
def fgsm_attack(model, data, target, epsilon):
    """
    实现FGSM攻击
    
    参数:
    model: 被攻击的模型
    data: 原始输入数据
    target: 真实标签
    epsilon: 扰动大小
    
    返回:
    perturbed_data: 对抗样本
    """
    # 设置模型为评估模式
    model.eval()
    
    # 设置输入数据需要梯度
    data.requires_grad = True
    
    # 前向传播
    output = model(data)
    loss = nn.functional.cross_entropy(output, target)
    
    # 反向传播，计算梯度
    model.zero_grad()
    loss.backward()
    
    # 获取输入数据的梯度
    data_grad = data.grad.data
    
    # 使用符号函数获取梯度的方向
    sign_data_grad = data_grad.sign()
    
    # 创建对抗样本
    perturbed_data = data + epsilon * sign_data_grad
    
    return perturbed_data

# 评估模型在FGSM攻击下的性能
def evaluate_fgsm(model, test_loader, epsilon):
    """
    评估模型在FGSM攻击下的性能
    
    参数:
    model: 被评估的模型
    test_loader: 测试数据加载器
    epsilon: 扰动大小
    
    返回:
    accuracy: 在对抗样本上的准确率
    examples: 示例样本
    """
    model.eval()
    correct = 0
    total = 0
    
    # 用于可视化
    examples = []
    
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        
        # 生成对抗样本
        perturbed_data = fgsm_attack(model, data, target, epsilon)
        
        # 预测
        output = model(perturbed_data)
        _, pred = output.max(1)
        
        total += target.size(0)
        correct += pred.eq(target).sum().item()
        
        # 保存一些示例用于可视化
        if len(examples) < 5:  # 保存5个示例
            examples.append((data.cpu(), perturbed_data.cpu(), target.cpu(), pred.cpu()))
    
    accuracy = 100. * correct / total
    print(f'Epsilon: {epsilon:.2f}, Test Accuracy under FGSM attack: {accuracy:.2f}%')
    
    return accuracy, examples

# 可视化对抗样本
def plot_adversarial_examples(examples, epsilon):
    """
    可视化原始图像和对抗样本
    
    参数:
    examples: 包含原始图像、对抗样本、真实标签和预测标签的列表
    epsilon: 扰动大小
    """
    plt.figure(figsize=(10, 8))
    for i, (original, adversarial, true_label, pred_label) in enumerate(examples):
        # 只取第一个样本
        original = original[0].squeeze()
        adversarial = adversarial[0].squeeze()
        true_label = true_label[0]
        pred_label = pred_label[0]
        
        # 计算扰动
        perturbation = adversarial - original
        
        # 绘制原始图像
        plt.subplot(5, 3, i*3+1)
        plt.imshow(original.detach().numpy(), cmap='gray')
        plt.title(f'Original: {true_label.item()}')
        plt.axis('off')
        
        # 绘制扰动
        plt.subplot(5, 3, i*3+2)
        # 将扰动标准化到[-1,1]范围以便可视化
        perturbation = (perturbation - perturbation.min()) / (perturbation.max() - perturbation.min())
        plt.imshow(perturbation.detach().numpy(), cmap='RdBu_r', vmin=0, vmax=1)
        plt.title(f'Perturbation (ε={epsilon})')
        plt.axis('off')
        
        # 绘制对抗样本
        plt.subplot(5, 3, i*3+3)
        plt.imshow(adversarial.detach().numpy(), cmap='gray')
        plt.title(f'Adversarial: {pred_label.item()}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'mnist_adversarial_examples_epsilon_{epsilon:.2f}.png')
    plt.show()

# 主函数
def main():
    # 加载测试数据
    test_loader = load_mnist_test()
    
    # 初始化模型
    model = MLP().to(device)
    
    # 加载预训练模型
    try:
        model.load_state_dict(torch.load('mnist_mlp_best.pth', map_location=device))
        print("预训练模型加载成功!")
    except FileNotFoundError:
        print("错误: 找不到预训练模型文件 'mnist_mlp_best.pth'")
        print("请确保模型文件存在于当前目录中")
        return
    
    # 评估模型在原始测试集上的性能
    print("评估模型在原始测试集上的性能...")
    test_accuracy = evaluate_model(model, test_loader)
    
    # FGSM攻击评估
    print("\n评估FGSM攻击...")
    epsilons = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7]
    accuracies = []
    all_examples = []
    
    for epsilon in epsilons:
        accuracy, examples = evaluate_fgsm(model, test_loader, epsilon)
        accuracies.append(accuracy)
        all_examples.append(examples)
        plot_adversarial_examples(all_examples[-1], epsilon)
    
    # 绘制准确率随epsilon变化的曲线
    plt.figure(figsize=(8, 6))
    plt.plot(epsilons, accuracies, '*-')
    plt.xlabel('Epsilon')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs Epsilon for FGSM Attack on MNIST')
    plt.grid(True)
    plt.savefig('mnist_accuracy_vs_epsilon.png')
    plt.show()
    
    # 可视化一些对抗样本 (选择epsilon=0.3的示例)
    plot_adversarial_examples(all_examples[-1], epsilons[-1])

if __name__ == "__main__":
    main()