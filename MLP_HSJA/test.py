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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------------------------
# 1. 加载 MNIST 测试数据
# ----------------------------
def load_mnist_test(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return test_loader

# ----------------------------
# 2. 定义 MLP 模型
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
# 4. HSJA-L2 攻击核心（MLP版）
# ----------------------------
@torch.no_grad()
def predict_label(model, x):
    return model(x).argmax(dim=1)

@torch.no_grad()
def is_adversarial(model, x, true_label):
    return (predict_label(model, x) != true_label).item()

@torch.no_grad()
def find_initial_adversarial(model, orig, true_label, max_trials=500, init_scale=1):
    B, C, H, W = orig.shape
    assert B == 1
    for t in range(max_trials):
        noise = torch.randn_like(orig) * init_scale
        candidate = orig + noise
        candidate = torch.clamp(candidate, -1.0, 1.0)
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
    adv = torch.clamp(adv, -1.0, 1.0)
    return adv

def hsja_attack_full(model, images, labels, max_iters=50, init_trials=500, q_min=20, q_max=100, sigma_init=1e-2, tol=1e-5, max_queries=5000):
    model.eval()
    B = images.size(0)
    adv_batch = images.clone()
    for idx in range(B):
        orig = images[idx:idx+1].to(device)
        true_label = labels[idx:idx+1].to(device)

        if is_adversarial(model, orig, true_label):
            adv_batch[idx:idx+1] = orig
            continue

        init_adv = find_initial_adversarial(model, orig, true_label, max_trials=init_trials)
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

            # 随机方向
            noises = torch.randn((q, *orig.shape[1:]), device=device)
            noises = noises.view(q, -1)
            noises = noises / (noises.norm(dim=1, keepdim=True) + 1e-12)
            noises = noises.view(q, *orig.shape[1:])

            grad_est = torch.zeros_like(current_adv)
            d0 = dist

            for i in range(q):
                pert = current_adv + sigma * noises[i:i+1]
                pert = torch.clamp(pert, -1.0, 1.0)
                d = torch.norm((pert - orig).view(-1), p=2).item()
                grad_est += ((d - d0)/sigma) * noises[i:i+1]
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
                candidate = torch.clamp(candidate, -1.0, 1.0)
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

def evaluate_hsja(model, test_loader, max_samples=50):
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
        orig_img = original.squeeze()
        adv_img = adversarial.squeeze()
        perturb = adv_img - orig_img

        plt.subplot(len(examples), 3, i*3+1)
        plt.imshow(orig_img.numpy(), cmap='gray')
        plt.title(f'Original: {true_label.item()}')
        plt.axis('off')

        plt.subplot(len(examples), 3, i*3+2)
        perturb = (perturb - perturb.min()) / (perturb.max() - perturb.min() + 1e-12)
        plt.imshow(perturb.numpy(), cmap='RdBu_r')
        plt.title('Perturb')
        plt.axis('off')

        plt.subplot(len(examples), 3, i*3+3)
        plt.imshow(adv_img.numpy(), cmap='gray')
        plt.title(f'Adversarial: {pred_label.item()}')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(f'adversarial_examples_{tag}.png')
    plt.show()

# ----------------------------
# 6. 主函数
# ----------------------------
def main():
    test_loader = load_mnist_test()
    model = MLP().to(device)

    try:
        model.load_state_dict(torch.load('mnist_mlp_best.pth', map_location=device))
        print("预训练模型加载成功!")
    except FileNotFoundError:
        print("找不到预训练模型文件 'mnist_mlp_best.pth'")
        return

    print("评估模型在原始测试集...")
    evaluate_model(model, test_loader)

    print("HSJA 攻击评估...")
    start_time = time.time()
    hsja_accuracy, examples = evaluate_hsja(model, test_loader, max_samples=20)
    end_time = time.time()
    print(f"HSJA evaluation time: {end_time - start_time:.1f}s")

    plot_adversarial_examples(examples, tag='hsja')

if __name__ == "__main__":
    main()
