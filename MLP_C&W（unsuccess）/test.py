# cw_attack_mlp.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from main import MLP  # 你的训练脚本中定义的模型类
import torch.nn.functional as F

# ---------- 配置 ----------
MODEL_PATH = "mnist_mlp_best.pth"
RESULT_IMG = "mnist_cw_results.png"
NUM_SAMPLES = 8   # 展示的样本数
SEED = 42

# C&W 超参数
INIT_C = 1e-3
KAPPA = 0
MAX_STEPS = 1000
LR = 0.01
BINARY_SEARCH_STEPS = 5
ABORT_EARLY = True

# ---------- 工具函数 ----------
def atanh(x):
    return 0.5 * torch.log((1 + x) / (1 - x + 1e-12) + 1e-12)

def denormalize(x_norm, mean, std):
    return x_norm * std.view(1,1,1) + mean.view(1,1,1)

def normalize(x, mean, std):
    return (x - mean.view(1,1,1)) / std.view(1,1,1)

def cw_loss_fn(logits, true_label, kappa=0, targeted=False, target_label=None):
    if targeted:
        assert target_label is not None
        target_logit = logits[0, target_label]
        others_no_target = torch.cat([logits[0,:target_label], logits[0,target_label+1:]])
        max_other = torch.max(others_no_target)
        f = torch.clamp(max_other - target_logit, min=-kappa)
    else:
        true_logit = logits[0, true_label]
        others_no_true = torch.cat([logits[0,:true_label], logits[0,true_label+1:]])
        max_other = torch.max(others_no_true)
        f = torch.clamp(max_other - true_logit, min=-kappa)
    return f

# ---------- 单次 C&W 优化 ----------
def cw_l2_single(model, x_norm, y, device, c, kappa=0, max_steps=1000, lr=0.01, targeted=False, target_label=None):
    mean = torch.tensor([0.1307]).to(device)
    std = torch.tensor([0.3081]).to(device)
    x_norm = x_norm.to(device)
    x_orig = denormalize(x_norm.squeeze(0), mean, std).unsqueeze(0).clamp(0,1)

    x_orig_t = (x_orig * 2.0) - 1.0
    x_orig_t = torch.clamp(x_orig_t, -0.999999, 0.999999)
    w_init = atanh(x_orig_t).detach()
    w = w_init.clone().detach().requires_grad_(True)
    optimizer = optim.Adam([w], lr=lr)

    prev_loss = 1e10
    patience = 0
    best_adv = None
    best_l2 = 1e10
    success = False

    for step in range(max_steps):
        optimizer.zero_grad()
        x_adv = 0.5 * (torch.tanh(w) + 1.0)
        x_adv_norm = normalize(x_adv.squeeze(0), mean, std).unsqueeze(0)
        logits = model(x_adv_norm)
        f_val = cw_loss_fn(logits, y, kappa=kappa, targeted=targeted, target_label=target_label)
        l2dist = torch.sum((x_adv - x_orig) ** 2)
        loss = l2dist + c * f_val
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            pred = torch.argmax(logits, dim=1).item()
            cur_l2 = l2dist.item()
            if (targeted and pred == target_label) or (not targeted and pred != y):
                success = True
                if cur_l2 < best_l2:
                    best_l2 = cur_l2
                    best_adv = x_adv.clone().detach()

        if ABORT_EARLY:
            if loss.item() > prev_loss - 1e-6:
                patience += 1
            else:
                patience = 0
            prev_loss = loss.item()
            if patience >= 100:
                break

    if best_adv is not None:
        final_adv = best_adv
    else:
        final_adv = x_adv.detach()

    final_adv_norm = normalize(final_adv.squeeze(0), mean, std).unsqueeze(0).detach()
    return final_adv_norm, success, best_l2

# ---------- 二分查找 c ----------
def cw_l2_binary_search_single(model, x_norm, y, device, init_c, binary_steps=5, **kwargs):
    c_low = 0.0
    c_high = 1e10
    c = init_c
    best_adv = None
    best_l2 = 1e10
    best_c = None
    success = False

    for _ in range(binary_steps):
        adv_norm, succ, l2 = cw_l2_single(model, x_norm, y, device, c, **kwargs)
        if succ:
            success = True
            if l2 < best_l2:
                best_l2 = l2
                best_adv = adv_norm
                best_c = c
            c_high = c
            c = (c_low + c_high) / 2 if c_high < 1e9 else c*0.5
        else:
            c_low = max(c, c_low)
            if c_high < 1e9:
                c = (c_low + c_high) / 2
            else:
                c *= 10
    if best_adv is None:
        return x_norm.clone(), False, None
    return best_adv, success, best_c

# ---------- 结果可视化 ----------
def visualize_results(origs, advs, orig_labels, adv_preds, adv_confs, savepath):
    n = len(origs)
    cols = 3
    rows = n
    plt.figure(figsize=(6 * cols, 2.5 * rows))
    mean = torch.tensor([0.1307])
    std = torch.tensor([0.3081])
    for i in range(n):
        plt.subplot(rows, cols, i*cols + 1)
        img = denormalize(origs[i].squeeze(0).cpu(), mean, std).squeeze(0).numpy()
        plt.imshow(img, cmap='gray')
        plt.title(f"orig: {orig_labels[i]}")
        plt.axis('off')

        plt.subplot(rows, cols, i*cols + 2)
        img_adv = denormalize(advs[i].squeeze(0).cpu(), mean, std).squeeze(0).numpy()
        plt.imshow(img_adv, cmap='gray')
        pred = adv_preds[i]
        title_color = 'green' if pred == orig_labels[i] else 'red'
        plt.title(f"adv pred: {pred}", color=title_color)
        plt.axis('off')

        plt.subplot(rows, cols, i*cols + 3)
        plt.axis('off')
        top3 = adv_confs[i]
        text = "\n".join([f"{lbl}: {prob:.4f}" for lbl, prob in top3])
        plt.text(0.01, 0.5, text, fontsize=14)
    plt.tight_layout()
    plt.savefig(savepath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Results saved to {savepath}")

# ---------- 主程序 ----------
def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = MLP().to(device)
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded model from {MODEL_PATH}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    indices = np.random.choice(len(test_set), NUM_SAMPLES, replace=False)

    origs, orig_labels, advs, adv_preds, adv_confs = [], [], [], [], []
    for i, idx in enumerate(indices):
        x_norm, y = test_set[idx]
        x_norm = x_norm.unsqueeze(0)
        origs.append(x_norm.clone())
        orig_labels.append(int(y))
        print(f"\nAttacking sample {i+1}/{len(indices)} (true={y})")

        adv_norm, success, used_c = cw_l2_binary_search_single(
            model, x_norm, y, device,
            init_c=INIT_C, binary_steps=BINARY_SEARCH_STEPS,
            kappa=KAPPA, max_steps=MAX_STEPS, lr=LR, targeted=False
        )
        with torch.no_grad():
            logits = model(adv_norm.to(device))
            pred = int(torch.argmax(logits, dim=1).item())
            probs = F.softmax(logits, dim=1)
            top3 = [(int(idx_), float(probs[0,idx_].cpu().item())) for idx_ in torch.topk(probs, 3, dim=1)[1][0]]
        advs.append(adv_norm.cpu())
        adv_preds.append(pred)
        adv_confs.append(top3)
        print(f"  success={success}, adv_pred={pred}, used_c={used_c}, top3={top3}")

    visualize_results(origs, advs, orig_labels, adv_preds, adv_confs, RESULT_IMG)

if __name__ == "__main__":
    main()
