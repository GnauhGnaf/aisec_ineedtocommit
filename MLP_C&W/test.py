import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
# 与训练代码保持一致的MLP模型结构
class MLP(nn.Module):
    """多层感知机模型 - 与训练代码保持完全一致"""
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

# ---------- 配置 ----------
MODEL_PATH = "mnist_mlp_best.pth"  # 确保模型文件存在
RESULT_IMG = "mnist_cw_results.png"
NUM_SAMPLES = 8   # 展示的样本数
SEED = 42

# C&W 超参数
INIT_C = 1e-3
KAPPA = 0.1       # 增加小值避免边界问题
MAX_STEPS = 1000  # 单轮优化步数
LR = 0.01         # 学习率
BINARY_SEARCH_STEPS = 5  # c 值二分搜索轮数
ABORT_EARLY = True       # 早停机制

# ---------- 工具函数 ----------
def atanh(x):
    """将 [-1,1] 区间值转换到 tanh 输入空间，避免数值溢出"""
    return 0.5 * torch.log((1 + x) / (1 - x + 1e-12) + 1e-12)

def denormalize(x_norm, mean, std):
    """反归一化：将归一化后的张量恢复为 [0,1] 图像（适配单通道 MNIST）"""
    return x_norm * std.view(1, 1, 1) + mean.view(1, 1, 1)

def normalize(x, mean, std):
    """归一化：将 [0,1] 图像转换为标准分布（适配单通道 MNIST）"""
    return (x - mean.view(1, 1, 1)) / std.view(1, 1, 1)

def cw_loss_fn(logits, true_label, kappa=0, targeted=False, target_label=None):
    """
    修复 C&W 损失函数符号：确保优化方向正确
    """
    if targeted:
        assert target_label is not None, "目标攻击需指定 target_label"
        target_logit = logits[0, target_label]
        # 排除目标标签后的其他类最大 logit
        mask = torch.ones_like(logits[0], dtype=torch.bool)
        mask[target_label] = False
        max_other = logits[0, mask].max()
        # f(x) = max_other - target_logit + kappa → 希望 f(x) ≤ 0（攻击成功）
        f_val = torch.clamp(max_other - target_logit + kappa, min=0)
    else:
        true_logit = logits[0, true_label]
        # 排除原标签后的其他类最大 logit
        mask = torch.ones_like(logits[0], dtype=torch.bool)
        mask[true_label] = False
        max_other = logits[0, mask].max()
        # f(x) = true_logit - max_other + kappa → 希望 f(x) ≤ 0（攻击成功）
        f_val = torch.clamp(true_logit - max_other + kappa, min=0)
    return f_val

# ---------- 单次 C&W 优化 ----------
def cw_l2_single(model, x_norm, y, device, c, kappa=0, max_steps=1000, lr=0.01, targeted=False, target_label=None):
    """
    单轮 C&W 攻击优化
    """
    # MNIST 归一化参数（固定值，与训练时一致）
    mean = torch.tensor([0.1307]).to(device)
    std = torch.tensor([0.3081]).to(device)
    
    # 初始化原始图像（反归一化到 [0,1]，转换到 tanh 输入空间 [-1,1]）
    x_norm = x_norm.to(device)  # (1,1,28,28)
    x_orig = denormalize(x_norm.squeeze(0), mean, std).unsqueeze(0).clamp(0, 1)  # (1,1,28,28)
    x_orig_t = (x_orig * 2.0) - 1.0  # 转换到 [-1,1]，适配 tanh
    x_orig_t = torch.clamp(x_orig_t, -0.999999, 0.999999)  # 避免 atanh 溢出
    
    # 初始化可优化变量 w（通过 tanh 映射回图像空间）
    w = atanh(x_orig_t).detach().requires_grad_(True)  # (1,1,28,28)，与 x_orig_t 同维度
    optimizer = optim.Adam([w], lr=lr)  # 优化 w 而非直接优化图像
    
    prev_loss = 1e10
    patience = 0
    best_adv = None
    best_l2 = 1e10
    success = False

    for step in range(max_steps):
        optimizer.zero_grad()
        
        # 1. 从 w 映射回 [0,1] 图像空间
        x_adv = 0.5 * (torch.tanh(w) + 1.0)  # (1,1,28,28) → [0,1]
        x_adv_norm = normalize(x_adv.squeeze(0), mean, std).unsqueeze(0)  # 归一化到训练分布
        
        # 2. 模型预测
        logits = model(x_adv_norm)  # 模型内部已通过Flatten处理维度
        
        # 3. 计算 C&W 损失（L2 距离 + c*惩罚项）
        f_val = cw_loss_fn(logits, y, kappa=kappa, targeted=targeted, target_label=target_label)
        l2_dist = torch.sum((x_adv - x_orig) ** 2)  # 对抗样本与原始样本的 L2 距离
        loss = l2_dist + c * f_val  # 总损失：最小化距离 + 确保攻击成功
        
        # 4. 反向传播与优化
        loss.backward()
        optimizer.step()

        # 5. 评估攻击效果（无梯度计算）
        with torch.no_grad():
            pred = torch.argmax(logits, dim=1).item()
            cur_l2 = l2_dist.item()
            
            # 攻击成功判断：无目标→预测≠原标签；有目标→预测=目标标签
            if (targeted and pred == target_label) or (not targeted and pred != y):
                success = True
                # 保留 L2 距离最小的对抗样本
                if cur_l2 < best_l2:
                    best_l2 = cur_l2
                    best_adv = x_adv_norm.clone().detach()  # 保存归一化后的对抗样本

        # 6. 早停机制
        if ABORT_EARLY:
            if loss.item() > prev_loss - 1e-6:  # 允许微小波动（1e-6）
                patience += 1
            else:
                patience = 0
            prev_loss = loss.item()
            if patience >= 100:  # 连续 100 步无下降则早停
                break

    # 若未找到成功的对抗样本，返回原始样本
    if best_adv is None:
        best_adv = x_adv_norm.detach()
        best_l2 = l2_dist.item()

    return best_adv, success, best_l2

# ---------- 二分查找最优 c 值 ----------
def cw_l2_binary_search_single(model, x_norm, y, device, init_c, binary_steps=5, **kwargs):
    """
    二分搜索最优惩罚系数 c
    """
    c_low = 0.0
    c_high = 1e10  # 初始上界（足够大）
    c = init_c
    best_adv = None
    best_l2 = 1e10
    best_c = None
    success = False

    for _ in range(binary_steps):
        # 单轮 c 值下的攻击
        adv_norm, succ, l2 = cw_l2_single(model, x_norm, y, device, c,** kwargs)
        
        if succ:
            # 攻击成功：更新最优结果，尝试更小的 c
            success = True
            if l2 < best_l2:
                best_l2 = l2
                best_adv = adv_norm
                best_c = c
            c_high = c  # 下一轮 c 上限设为当前 c
            # 调整 c：若上界未达阈值，取中间值；否则减半
            c = (c_low + c_high) / 2 if c_high < 1e9 else c * 0.5
        else:
            # 攻击失败：增大 c，提高惩罚力度
            c_low = max(c, c_low)  # 下界设为当前 c（确保递增）
            # 调整 c：若上界未达阈值，取中间值；否则乘 10
            c = (c_low + c_high) / 2 if c_high < 1e9 else c * 10

    # 若所有 c 都失败，返回原始样本
    if best_adv is None:
        return x_norm.clone(), False, None

    return best_adv, success, best_c

# ---------- 结果可视化 ----------
def visualize_results(origs, advs, orig_labels, adv_preds, adv_confs, savepath):
    """
    可视化原始样本、对抗样本、Top3 置信度
    """
    n = len(origs)
    cols = 3  # 每列：原始样本、对抗样本、置信度
    rows = n
    plt.figure(figsize=(6 * cols, 2.5 * rows))  # 适配 8 个样本的画布大小
    mean = torch.tensor([0.1307])  # MNIST 均值
    std = torch.tensor([0.3081])  # MNIST 标准差

    for i in range(n):
        # 1. 绘制原始样本
        plt.subplot(rows, cols, i * cols + 1)
        orig_img = denormalize(origs[i].squeeze(0).cpu(), mean, std).squeeze(0).numpy()
        plt.imshow(orig_img, cmap='gray')
        plt.title(f"Original: {orig_labels[i]}", fontsize=12)
        plt.axis('off')

        # 2. 绘制对抗样本
        plt.subplot(rows, cols, i * cols + 2)
        adv_img = denormalize(advs[i].squeeze(0).cpu(), mean, std).squeeze(0).numpy()
        plt.imshow(adv_img, cmap='gray')
        title_color = 'green' if adv_preds[i] == orig_labels[i] else 'red'
        plt.title(f"Adv Pred: {adv_preds[i]}", color=title_color, fontsize=12)
        plt.axis('off')

        # 3. 绘制 Top3 置信度
        plt.subplot(rows, cols, i * cols + 3)
        plt.axis('off')
        top3 = adv_confs[i]
        text = "\n".join([f"Class {lbl}: {prob:.4f}" for lbl, prob in top3])
        plt.text(0.01, 0.5, text, fontsize=11, verticalalignment='center')

    plt.tight_layout()
    plt.savefig(savepath, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"攻击结果已保存到：{savepath}")

# ---------- 主程序 ----------
def main():
    # 1. 固定随机种子
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # 2. 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}")

    # 3. 加载 MLP 模型（与训练代码结构一致）
    model = MLP().to(device)
    try:
        # 加载预训练模型权重
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()  # 攻击时禁用 dropout/batchnorm 更新
        print(f"成功加载模型：{MODEL_PATH}")
    except FileNotFoundError:
        print(f"错误：未找到模型文件 {MODEL_PATH}，请先训练 MLP 模型！")
        return

    # 4. 加载 MNIST 测试集
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转换为 (1,28,28) 张量，值 [0,1]
        transforms.Normalize((0.1307,), (0.3081,))  # 归一化到标准分布
    ])
    test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    # 随机选择 NUM_SAMPLES 个测试样本
    indices = np.random.choice(len(test_set), NUM_SAMPLES, replace=False)

    # 5. 存储攻击结果
    origs = []       # 原始样本（归一化后）
    orig_labels = [] # 原始标签
    advs = []        # 对抗样本（归一化后）
    adv_preds = []   # 对抗样本预测标签
    adv_confs = []   # 对抗样本 Top3 置信度

    # 6. 逐个样本执行 C&W 攻击
    for i, idx in enumerate(indices):
        x_norm, y = test_set[idx]  # x_norm: (1,28,28)，y: 原始标签（int）
        x_norm = x_norm.unsqueeze(0)  # 扩展为 batch_size=1：(1,1,28,28)
        
        # 记录原始样本信息
        origs.append(x_norm.clone())
        orig_labels.append(int(y))
        print(f"\n正在攻击第 {i+1}/{NUM_SAMPLES} 个样本（原始标签：{y}）")

        # 执行 C&W 攻击（二分搜索最优 c）
        adv_norm, success, used_c = cw_l2_binary_search_single(
            model, x_norm, y, device,
            init_c=INIT_C,
            binary_steps=BINARY_SEARCH_STEPS,
            kappa=KAPPA,
            max_steps=MAX_STEPS,
            lr=LR,
            targeted=False  # 无目标攻击：仅需改变预测结果
        )

        # 分析对抗样本结果
        with torch.no_grad():
            logits = model(adv_norm.to(device))
            pred = int(torch.argmax(logits, dim=1).item())  # 预测标签
            probs = F.softmax(logits, dim=1)  # 转换为置信度
            # 获取 Top3 置信度的类别与概率
            top3_indices = torch.topk(probs, 3, dim=1)[1][0]
            top3 = [(int(idx), float(probs[0, idx].cpu().item())) for idx in top3_indices]

        # 记录对抗样本信息
        advs.append(adv_norm.cpu())
        adv_preds.append(pred)
        adv_confs.append(top3)

        # 打印攻击结果
        success_str = "✓ 成功" if success else "✗ 失败"
        print(f"  攻击结果：{success_str} | 对抗预测：{pred} | 使用 c 值：{used_c:.4f} | Top3 置信度：{top3}")

    # 7. 可视化所有攻击结果
    visualize_results(origs, advs, orig_labels, adv_preds, adv_confs, RESULT_IMG)

if __name__ == "__main__":
    main()
