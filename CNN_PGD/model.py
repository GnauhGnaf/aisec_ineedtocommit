import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# 设置随机种子确保可重复性
torch.manual_seed(42)
np.random.seed(42)

# 检查是否有可用的GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. 数据加载与预处理
def load_cifar10():
    # 定义数据预处理
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    # 加载CIFAR-10数据集
    train_set = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)

    # 创建数据加载器
    train_loader = DataLoader(
        train_set, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(
        test_set, batch_size=128, shuffle=False, num_workers=2)
    
    return train_loader, test_loader

# 2. 定义卷积神经网络模型
class CIFAR10CNN(nn.Module):
    def __init__(self):
        super(CIFAR10CNN, self).__init__()
        self.features = nn.Sequential(
            # 输入: 3x32x32
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出: 64x16x16
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出: 128x8x8
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 输出: 256x4x4
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 10),  # 输出层为10个神经元
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# 3. 评估函数
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

# 4+. PGD攻击实现
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
        original = original[0]
        adversarial = adversarial[0]
        true_label = true_label[0]
        pred_label = pred_label[0]
        
        # 反归一化
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
        std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
        original = original * std + mean
        adversarial = adversarial * std + mean
        
        # 计算扰动
        perturbation = adversarial - original
        
        # 绘制原始图像
        plt.subplot(5, 3, i*3+1)
        plt.imshow(np.transpose(original.detach().numpy(), (1, 2, 0)))
        plt.title(f'Original: {true_label.item()}')
        plt.axis('off')
        
        # 绘制扰动
        plt.subplot(5, 3, i*3+2)
        # 将扰动标准化到[0,1]范围以便可视化
        perturbation = (perturbation - perturbation.min()) / (perturbation.max() - perturbation.min())
        plt.imshow(np.transpose(perturbation.detach().numpy(), (1, 2, 0)))
        plt.title(f'Perturbation (ε={epsilon})')
        plt.axis('off')
        
        # 绘制对抗样本
        plt.subplot(5, 3, i*3+3)
        plt.imshow(np.transpose(adversarial.detach().numpy(), (1, 2, 0)))
        plt.title(f'Adversarial: {pred_label.item()}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'adversarial_examples_epsilon_{epsilon}.png')
    plt.show()

# 5. 主函数
def main():
    # 加载数据
    train_loader, test_loader = load_cifar10()
    
    # 初始化模型
    model = CIFAR10CNN().to(device)
    
    # 加载预训练模型
    try:
        model.load_state_dict(torch.load('cifar10_cnn.pth', map_location=device))
        print("预训练模型加载成功!")
    except FileNotFoundError:
        print("错误: 找不到预训练模型文件 'cifar10_cnn.pth'")
        print("请确保模型文件存在于当前目录中")
        return
    
    # 评估模型
    print("评估模型在原始测试集上的性能...")
    test_accuracy = evaluate_model(model, test_loader)
    
        # PGD攻击评估
    print("评估PGD攻击...")
    epsilons = [0,0.01, 0.05, 0.1, 0.2]
    alpha = 0.01  # 每步步长
    iters = 10    # 迭代次数
    pgd_accuracies = []

    for epsilon in epsilons:
        accuracy, _ = evaluate_pgd(model, test_loader, epsilon, alpha, iters)
        pgd_accuracies.append(accuracy)

    # 绘制 PGD 曲线
    plt.figure(figsize=(8, 6))
    plt.plot(epsilons, pgd_accuracies, 'o-', label='PGD')
    plt.xlabel('Epsilon')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs Epsilon for PGD Attack')
    plt.grid(True)
    plt.legend()
    plt.savefig('pgd_accuracy_vs_epsilon.png')
    plt.show()


if __name__ == "__main__":
    main()