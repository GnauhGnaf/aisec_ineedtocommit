import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import time

# ----------------------------
# 设备和随机种子
# ----------------------------
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------------------------
# 1. 数据加载
# ----------------------------
def load_cifar10():
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
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False, num_workers=2)
    return train_loader, test_loader

# ----------------------------
# 2. CNN 模型
# ----------------------------
class CIFAR10CNN(nn.Module):
    def __init__(self):
        super(CIFAR10CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256*4*4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# ----------------------------
# 3. 评估函数
# ----------------------------
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
    accuracy = 100.0 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy

# ----------------------------
# 4. 针对敏感像素的比特翻转攻击
# ----------------------------
@torch.no_grad()
def predict_label(model, x):
    return model(x).argmax(dim=1)

# @torch.no_grad()
def bit_flip_attack(model, images, labels, max_flips=10, epsilon=1e-2, top_k=3):
    """
    优化版针对敏感像素的比特翻转攻击（真实梯度敏感度）
    每次翻转 top_k 个最敏感像素
    """
    model.eval()
    adv_batch = images.clone()
    B, C, H, W = images.shape

    for idx in range(B):
        orig = images[idx:idx+1].to(device)
        true_label = labels[idx:idx+1].to(device)

        # 如果原始样本已被错分类，直接跳过
        if predict_label(model, orig).item() != true_label.item():
            adv_batch[idx:idx+1] = orig
            print(f"[BitFlip-Opt] idx={idx}: 原始样本已错分类")
            continue

        adv = orig.clone()
        flipped_positions = set()

        for flip_idx in range(max_flips):
            adv = adv.clone().detach()
            adv.requires_grad = True

            # ----------------------------
            # 1. 计算像素敏感度（梯度绝对值）
            # ----------------------------
            output = model(adv)
            prob = torch.softmax(output, dim=1)[0, true_label]
            model.zero_grad()
            prob.backward()
            grads = adv.grad.data.abs()[0]  # (C,H,W)

            # 将已经翻转过的像素敏感度设为 -1，避免重复选择
            for c, h, w in flipped_positions:
                grads[c, h, w] = -1.0

            # ----------------------------
            # 2. 找 top_k 最敏感像素并翻转
            # ----------------------------
            flat_idx = torch.argsort(grads.flatten(), descending=True)[:top_k]
            for f in flat_idx:
                c_max, h_max, w_max = np.unravel_index(f.cpu().numpy(), (C, H, W))
                val = adv[0, c_max, h_max, w_max] * -1.0  # 非原地操作
                adv = adv.clone()  # clone 避免叶子张量原地报错
                adv[0, c_max, h_max, w_max] = val
                flipped_positions.add((c_max, h_max, w_max))

            adv = adv.detach()
            adv.requires_grad = False

            # ----------------------------
            # 3. 检查是否成功
            # ----------------------------
            if predict_label(model, adv).item() != true_label.item():
                print(f"[BitFlip-Opt] idx={idx}: 成功生成对抗样本 after {flip_idx+1} flips")
                break

        adv_batch[idx:idx+1] = adv
        if predict_label(model, adv).item() == true_label.item():
            print(f"[BitFlip-Opt] idx={idx}: 攻击失败 after {max_flips} flips")

    return adv_batch



def evaluate_bitflip(model, test_loader, max_samples=5, max_flips=10):
    """
    在测试集上评估比特翻转攻击，只攻击前 max_samples 个样本，可增加 max_flips
    """
    model.eval()
    correct = 0
    total = 0
    examples = []
    processed = 0
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        B = data.size(0)
        if processed + B > max_samples:
            B = max_samples - processed
            data = data[:B]
            target = target[:B]
        adv = bit_flip_attack(model, data, target, max_flips=max_flips)
        output = model(adv)
        _, pred = output.max(1)
        total += target.size(0)
        correct += pred.eq(target).sum().item()
        for i in range(adv.size(0)):
            examples.append((data[i].cpu(), adv[i].cpu(), target[i].cpu(), pred[i].cpu()))
        processed += B
        if processed >= max_samples:
            break
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    print(f"Bit-Flip-Sens Test Accuracy (first {processed} samples): {accuracy:.2f}%")
    return accuracy, examples

# ----------------------------
# 5. 可视化
# ----------------------------
def plot_adversarial_examples(examples, tag='bitflip_sens'):
    if len(examples) == 0:
        print("No examples to plot.")
        return
    plt.figure(figsize=(10, 8))
    for i, (original, adversarial, true_label, pred_label) in enumerate(examples):
        mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3,1,1)
        std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3,1,1)
        orig_img = original * std + mean
        adv_img = adversarial * std + mean
        perturb = adv_img - orig_img

        plt.subplot(len(examples), 3, i*3+1)
        plt.imshow(np.transpose(orig_img.numpy(), (1,2,0)))
        plt.title(f'Original: {true_label.item()}')
        plt.axis('off')

        plt.subplot(len(examples), 3, i*3+2)
        p = (perturb - perturb.min()) / (perturb.max() - perturb.min() + 1e-12)
        plt.imshow(np.transpose(p.numpy(), (1,2,0)))
        plt.title('Perturb')
        plt.axis('off')

        plt.subplot(len(examples), 3, i*3+3)
        plt.imshow(np.transpose(adv_img.numpy(), (1,2,0)))
        plt.title(f'Adversarial: {pred_label.item()}')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'adversarial_examples_{tag}.png')
    plt.show()

# ----------------------------
# 6. 主函数
# ----------------------------
def main():
    train_loader, test_loader = load_cifar10()
    model = CIFAR10CNN().to(device)
    try:
        model.load_state_dict(torch.load('cifar10_cnn.pth', map_location=device))
        print("预训练模型加载成功!")
    except FileNotFoundError:
        print("找不到预训练模型文件 'cifar10_cnn.pth'")
        return

    # print("评估模型在原始测试集...")
    # evaluate_model(model, test_loader)

    print("比特翻转攻击评估 (仅前5个样本)...")
    start_time = time.time()
    # 攻击前5个样本，最多翻转10个像素
    bf_accuracy, examples = evaluate_bitflip(model, test_loader, max_samples=5, max_flips=1000000)
    end_time = time.time()
    print(f"Bit-Flip-Sens evaluation time: {end_time - start_time:.1f}s")

    plot_adversarial_examples(examples, tag='bitflip_sens')

if __name__ == "__main__":
    main()
