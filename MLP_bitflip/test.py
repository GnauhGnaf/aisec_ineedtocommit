import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
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
def load_mnist(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

# ----------------------------
# 2. MLP 模型
# ----------------------------
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

# ----------------------------
# 3. 评估函数
# ----------------------------
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    accuracy = 100.0 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy

# ----------------------------
# 4. 针对黑白像素的比特翻转攻击 (MLP)
# ----------------------------
@torch.no_grad()
def predict_label(model, x):
    return model(x).argmax(dim=1)

def bit_flip_attack(model, images, labels, max_flips=10, top_k=3):
    """
    对 MNIST MLP 的黑白像素进行比特翻转攻击
    每次翻转 top_k 个未翻转的像素，像素值严格翻转 0<->1
    """
    model.eval()
    adv_batch = images.clone()
    B, C, H, W = images.shape
    N = C*H*W  # 展平维度

    for idx in range(B):
        orig = images[idx:idx+1].to(device)
        true_label = labels[idx:idx+1].to(device)

        if predict_label(model, orig).item() != true_label.item():
            adv_batch[idx:idx+1] = orig
            print(f"[BitFlip-BW] idx={idx}: 原始样本已错分类")
            continue

        adv = orig.clone()
        adv_flat = adv.view(1, -1)
        flipped_positions = set()

        for flip_idx in range(max_flips):
            # 1. 计算每个像素对概率的敏感度
            prob_orig = torch.softmax(model(adv_flat), dim=1)[0, true_label]
            grad_estimate = torch.zeros(N)
            for i in range(N):
                if i in flipped_positions:
                    grad_estimate[i] = -1  # 已翻转的不再选
                else:
                    tmp = adv_flat.clone()
                    # 翻转该位 0<->1
                    tmp[0, i] = 1.0 - tmp[0, i]
                    prob_tmp = torch.softmax(model(tmp), dim=1)[0, true_label]
                    grad_estimate[i] = (prob_orig - prob_tmp).item()

            # 2. 选择 top_k 最大敏感像素进行翻转
            topk_idx = torch.argsort(grad_estimate, descending=True)[:top_k]
            for f in topk_idx:
                adv_flat[0, f] = 1.0 - adv_flat[0, f]  # 黑白翻转
                flipped_positions.add(f.item())

            # 3. 检查是否成功
            if predict_label(model, adv_flat.view(1, C, H, W)).item() != true_label.item():
                print(f"[BitFlip-BW] idx={idx}: 成功生成对抗样本 after {len(flipped_positions)} flips")
                break

            if len(flipped_positions) >= max_flips:
                break

        adv_batch[idx:idx+1] = adv_flat.view(1, C, H, W)
        if predict_label(model, adv_flat.view(1, C, H, W)).item() == true_label.item():
            print(f"[BitFlip-BW] idx={idx}: 攻击失败 after {len(flipped_positions)} flips")

    return adv_batch

def evaluate_bitflip(model, test_loader, max_samples=5, max_flips=100):
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
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)

        for i in range(B):
            examples.append((data[i].cpu(), adv[i].cpu(), target[i].cpu(), pred[i].cpu()))
        processed += B
        if processed >= max_samples:
            break

    accuracy = 100.0 * correct / total if total > 0 else 0.0
    print(f"Bit-Flip-BW Test Accuracy (first {processed} samples): {accuracy:.2f}%")
    return accuracy, examples

# ----------------------------
# 5. 可视化
# ----------------------------
def plot_adversarial_examples(examples, tag='bitflip_bw'):
    if len(examples) == 0:
        print("No examples to plot.")
        return
    plt.figure(figsize=(10, 8))
    for i, (original, adversarial, true_label, pred_label) in enumerate(examples):
        mean = torch.tensor([0.1307]).view(1,1,1)
        std = torch.tensor([0.3081]).view(1,1,1)
        orig_img = original * std + mean
        adv_img = adversarial * std + mean
        perturb = adv_img - orig_img

        plt.subplot(len(examples), 3, i*3+1)
        plt.imshow(orig_img.squeeze().numpy(), cmap='gray')
        plt.title(f'Original: {true_label.item()}')
        plt.axis('off')

        plt.subplot(len(examples), 3, i*3+2)
        plt.imshow((perturb.squeeze().numpy() != 0).astype(float), cmap='gray')
        plt.title('Perturb')
        plt.axis('off')

        plt.subplot(len(examples), 3, i*3+3)
        plt.imshow(adv_img.squeeze().numpy(), cmap='gray')
        plt.title(f'Adversarial: {pred_label.item()}')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(f'adversarial_examples_{tag}.png')
    plt.show()

# ----------------------------
# 6. 主函数
# ----------------------------
def main():
    train_loader, test_loader = load_mnist()
    model = MLP().to(device)

    try:
        model.load_state_dict(torch.load('mnist_mlp_best.pth', map_location=device))
        print("预训练模型加载成功!")
    except FileNotFoundError:
        print("找不到预训练模型文件 'mnist_mlp_best.pth'")
        return

    print("评估模型在原始测试集...")
    evaluate_model(model, test_loader)

    print("比特翻转攻击评估 (仅前5个样本)...")
    start_time = time.time()
    bf_accuracy, examples = evaluate_bitflip(model, test_loader, max_samples=5, max_flips=100)
    end_time = time.time()
    print(f"Bit-Flip-BW evaluation time: {end_time - start_time:.1f}s")

    plot_adversarial_examples(examples, tag='bitflip_bw')

if __name__ == "__main__":
    main()
