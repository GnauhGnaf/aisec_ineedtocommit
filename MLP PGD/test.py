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

def pgd_attack(model, data, target, epsilon, alpha, iters):
    """
    实现PGD攻击 (L∞ 范数约束)
    
    参数:
    model: 被攻击的模型
    data: 原始输入数据
    target: 真实标签
    epsilon: 扰动大小
    alpha: 每步迭代步长
    iters: 迭代次数
    
    返回:
    perturbed_data: 对抗样本
    """
    # 初始化为原始图像 + 随机噪声（可以注释掉这行禁用随机初始化）
    perturbed_data = data.clone().detach()
    perturbed_data = torch.clamp(perturbed_data, 0, 1)  # 保证在合法范围
    perturbed_data.requires_grad = True

    for _ in range(iters):
        output = model(perturbed_data)
        loss = nn.functional.cross_entropy(output, target)

        model.zero_grad()
        loss.backward()
        
        # 梯度上升一小步
        with torch.no_grad():
            perturbed_data = perturbed_data + alpha * perturbed_data.grad.sign()
            # 投影回 L∞ 球内
            perturbation = torch.clamp(perturbed_data - data, min=-epsilon, max=epsilon)
            perturbed_data = data + perturbation

        perturbed_data.requires_grad = True

    return perturbed_data.detach()

def evaluate_pgd(model, test_loader, epsilon, alpha, iters):
    """
    评估模型在PGD攻击下的性能
    """
    model.eval()
    correct = 0
    total = 0
    examples = []

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        
        perturbed_data = pgd_attack(model, data, target, epsilon, alpha, iters)
        output = model(perturbed_data)
        _, pred = output.max(1)

        total += target.size(0)
        correct += pred.eq(target).sum().item()

        if len(examples) < 5:
            examples.append((data.cpu(), perturbed_data.cpu(), target.cpu(), pred.cpu()))

    accuracy = 100. * correct / total
    print(f"Epsilon: {epsilon}, PGD Attack Accuracy: {accuracy:.2f}%")

    plot_adversarial_examples(examples, epsilon)
    return accuracy, examples

# 可视化对抗样本（兼容灰度）
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
    plt.savefig(f'mnist_PGD_examples_epsilon_{epsilon:.2f}.png')
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
    _ = evaluate_model(model, test_loader)
    
    print("\n评估PGD攻击...")
    epsilons = [0, 0.05, 0.1, 0.2, 0.3]
    alpha = 0.1  # 每步步长（与 CIFAR 代码保持一致风格）
    iters = 10    # 迭代次数
    pgd_accuracies = []

    for epsilon in epsilons:
        acc, _ = evaluate_pgd(model, test_loader, epsilon, alpha, iters)
        pgd_accuracies.append(acc)

    # 绘制 PGD 准确率曲线
    plt.figure(figsize=(8, 6))
    plt.plot(epsilons, pgd_accuracies, 'o-', label='PGD')
    plt.xlabel('Epsilon')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs Epsilon for PGD Attack on MNIST (MLP)')
    plt.grid(True)
    plt.legend()
    plt.savefig('mnist_pgd_accuracy_vs_epsilon.png')
    plt.show()

if __name__ == "__main__":
    main()
