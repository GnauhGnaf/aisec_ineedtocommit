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
# 4. HSJA-L2 攻击核心函数（完整论文版）
# ----------------------------
@torch.no_grad()
def predict_label(model, x):
    return model(x).argmax(dim=1)

@torch.no_grad()
def is_adversarial(model, x, true_label):
    return (predict_label(model, x) != true_label).item()

@torch.no_grad()
def find_initial_adversarial(model, orig, true_label, max_trials=500, init_scale=0.5):
    B, C, H, W = orig.shape
    assert B == 1
    for t in range(max_trials):
        noise = torch.randn_like(orig) * init_scale
        candidate = orig + noise
        candidate = torch.clamp(candidate, -5.0, 5.0)
        if is_adversarial(model, candidate, true_label):
            return candidate
    return None

@torch.no_grad()
def binary_search_boundary(model, orig, adv_candidate, true_label, tol=1e-5, max_steps=20):
    low = orig.clone()
    high = adv_candidate.clone()
    if not is_adversarial(model, high, true_label):
        return None
    for _ in range(max_steps):
        mid = (low + high) / 2.0
        if is_adversarial(model, mid, true_label):
            high = mid
        else:
            low = mid
        if torch.norm((high - low).view(-1)) < tol:
            break
    return high

@torch.no_grad()
def project_onto_sphere(orig, direction, radius):
    adv = orig + radius * direction
    adv = torch.clamp(adv, -5.0, 5.0)
    return adv

def hsja_attack_full(model, images, labels, max_iters=50, init_trials=5000, q_min=20, q_max=100, sigma_init=1e-2, tol=1e-5, max_queries=5000):
    model.eval()
    B = images.size(0)
    adv_batch = images.clone()
    for idx in range(B):
        orig = images[idx:idx+1].to(device)
        true_label = labels[idx:idx+1].to(device)
        if is_adversarial(model, orig, true_label):
            adv_batch[idx:idx+1] = orig
            continue
        init_adv = find_initial_adversarial(model, orig, true_label, max_trials=init_trials, init_scale=1)
        if init_adv is None:
            print(f"[HSJA] idx={idx}: 初始对抗失败")
            adv_batch[idx:idx+1] = orig
            continue
        boundary = binary_search_boundary(model, orig, init_adv, true_label, max_steps=20)
        if boundary is None:
            adv_batch[idx:idx+1] = orig
            continue
        current_adv = boundary.clone()
        dist = torch.norm((current_adv - orig).view(-1), p=2).item()
        sigma = sigma_init
        query_count = 0

        for it in range(max_iters):
            q = int(q_min + (q_max - q_min) * (it / max_iters))
            shape = current_adv.shape
            noises = torch.randn((q, *shape[1:]), device=device)
            noises = noises / noises.view(q, -1).norm(dim=1).view(q,1,1,1)
            grad_est = torch.zeros_like(current_adv)
            d0 = dist
            for i in range(q):
                pert = current_adv + sigma * noises[i:i+1]
                pert = torch.clamp(pert, -5.0, 5.0)
                adv_flag = is_adversarial(model, pert, true_label)
                d = torch.norm((pert - orig).view(-1), p=2).item()
                grad_est += ((d - d0) / sigma) * noises[i:i+1]
                query_count += 1
                if query_count >= max_queries:
                    break
            if query_count >= max_queries:
                break
            grad_est = -grad_est
            grad_est = grad_est / (torch.norm(grad_est.view(-1), p=2).item() + 1e-12)

            step_size = 0.1 * dist
            success = False
            for _ in range(10):
                candidate = current_adv + step_size * grad_est
                candidate = project_onto_sphere(orig, candidate - orig, dist)
                candidate = torch.clamp(candidate, -5.0, 5.0)
                candidate_boundary = binary_search_boundary(model, orig, candidate, true_label, max_steps=10)
                if candidate_boundary is not None:
                    new_dist = torch.norm((candidate_boundary - orig).view(-1), p=2).item()
                    if new_dist < dist:
                        current_adv = candidate_boundary
                        dist = new_dist
                        success = True
                        break
                step_size *= 0.5
            if not success or step_size < tol or dist < tol:
                break
        adv_batch[idx:idx+1] = current_adv
        print(f"[HSJA] idx={idx}: finished. L2={dist:.4f}, queries={query_count}")
    return adv_batch

def evaluate_hsja(model, test_loader, max_samples=100):
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
        adv = hsja_attack_full(model, data, target, max_iters=50, init_trials=500)
        output = model(adv)
        _, pred = output.max(1)
        total += target.size(0)
        correct += pred.eq(target).sum().item()
        for i in range(min(5 - len(examples), adv.size(0))):
            examples.append((data[i].cpu(), adv[i].cpu(), target[i].cpu(), pred[i].cpu()))
        processed += B
        if processed >= max_samples:
            break
    accuracy = 100.0 * correct / total if total > 0 else 0.0
    print(f"HSJA Test Accuracy (first {processed} samples): {accuracy:.2f}%")
    return accuracy, examples

# ----------------------------
# 5. 可视化
# ----------------------------
def plot_adversarial_examples(examples, tag='hsja'):
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

    print("评估模型在原始测试集...")
    evaluate_model(model, test_loader)

    print("HSJA 攻击评估...")
    start_time = time.time()
    hsja_accuracy, examples = evaluate_hsja(model, test_loader, max_samples=50)  # 测试前50个样本
    end_time = time.time()
    print(f"HSJA evaluation time: {end_time - start_time:.1f}s")

    plot_adversarial_examples(examples, tag='hsja')

if __name__ == "__main__":
    main()
