import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# GPU 检查
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. 数据加载与预处理
def load_cifar10():
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    test_set = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    return test_loader

# 2. CNN模型
class CIFAR10CNN(nn.Module):
    def __init__(self):
        super(CIFAR10CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(64,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(128,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,3,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256*4*4,1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024,512),
            nn.ReLU(inplace=True),
            nn.Linear(512,10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# 3. 可视化函数
def plot_adversarial_examples(examples, confidences, perturbation_scale=0.1, save_path='adversarial_examples.png'):
    """
    examples: list of (original, adversarial, true_label)
    confidences: list of dict，每张样本的10类预测概率
    perturbation_scale: 扰动缩放系数，越小差异越不明显
    save_path: 保存路径
    """
    import matplotlib.gridspec as gridspec

    mean = torch.tensor([0.4914,0.4822,0.4465]).view(3,1,1)
    std = torch.tensor([0.2470,0.2435,0.2616]).view(3,1,1)
    
    num_samples = len(examples)
    fig = plt.figure(figsize=(12, 4*num_samples))
    gs = gridspec.GridSpec(num_samples, 4, width_ratios=[1,1,1,0.5])

    for i, (original, adversarial, true_label) in enumerate(examples):
        original_img = original[0]*std + mean
        adv_img = adversarial[0]*std + mean
        perturbation = adv_img - original_img
        adv_img_scaled = torch.clamp(original_img + perturbation*perturbation_scale, 0, 1)
        perturb_norm = (perturbation - perturbation.min()) / (perturbation.max()-perturbation.min())
        
        # 原图
        ax0 = fig.add_subplot(gs[i,0])
        ax0.imshow(np.transpose(original_img.detach().numpy(),(1,2,0)))
        ax0.set_title(f"Original: {true_label}", fontsize=10)
        ax0.axis('off')
        
        # 扰动
        ax1 = fig.add_subplot(gs[i,1])
        ax1.imshow(np.transpose(perturb_norm.detach().numpy(),(1,2,0)))
        ax1.set_title("Perturbation", fontsize=10)
        ax1.axis('off')
        
        # 对抗样本
        ax2 = fig.add_subplot(gs[i,2])
        ax2.imshow(np.transpose(adv_img_scaled.detach().numpy(),(1,2,0)))
        ax2.set_title("Adversarial", fontsize=10)
        ax2.axis('off')
        
        # 置信度显示在右侧单独子图
        ax3 = fig.add_subplot(gs[i,3])
        ax3.axis('off')
        conf = confidences[i]
        top3 = sorted(conf.items(), key=lambda x:x[1], reverse=True)[:3]
        conf_text = "\n".join([f"Class {k}: {v:.2f}" for k,v in top3])
        ax3.text(0, 0.5, conf_text, fontsize=10, va='center', ha='left')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"保存图片到 {save_path}")



# 4. C&W攻击（每张图像二分搜索c）
def cw_attack_binary_search_single(model, data, target, c_low=1e-3, c_high=1e10, lr=0.01, steps=100, max_binary_steps=10):
    model.eval()
    data = data.clone().detach().to(device)
    target = target.clone().detach().to(device)
    best_adv = data.clone()
    
    for _ in range(max_binary_steps):
        c = (c_low + c_high)/2
        perturbation = torch.zeros_like(data, requires_grad=True).to(device)
        optimizer = optim.Adam([perturbation], lr=lr)
        
        for _ in range(steps):
            adv_data = torch.clamp(data + perturbation,0,1)
            outputs = model(adv_data)
            one_hot = torch.nn.functional.one_hot(target,10).float()
            real = torch.sum(one_hot*outputs,dim=1)
            other,_ = torch.max((1-one_hot)*outputs - one_hot*1e4,dim=1)
            f_loss = torch.clamp(real-other,min=0)
            l2_loss = torch.sum((adv_data-data)**2,dim=[1,2,3])
            loss = torch.mean(l2_loss + c*f_loss)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        adv_data = torch.clamp(data + perturbation.detach(),0,1)
        pred = model(adv_data).argmax(dim=1)
        success = ~pred.eq(target)
        if success.item():
            c_high = c
            best_adv = adv_data
        else:
            c_low = c
            
    return best_adv

# 5. 主函数
def main():
    test_loader = load_cifar10()
    model = CIFAR10CNN().to(device)
    try:
        model.load_state_dict(torch.load('cifar10_cnn.pth', map_location=device))
        print("预训练模型加载成功!")
    except FileNotFoundError:
        print("找不到模型文件 'cifar10_cnn.pth'")
        return
    
    # 只取前5个样本进行攻击
    examples = []
    confidences = []
    count = 0
    for data,target in test_loader:
        data, target = data.to(device), target.to(device)
        adv_data = cw_attack_binary_search_single(model, data, target)
        output = model(adv_data)
        probs = torch.softmax(output[0],dim=0)
        conf_dict = {i: probs[i].item() for i in range(10)}
        examples.append((data.cpu(), adv_data.cpu(), target.item()))
        confidences.append(conf_dict)
        count += 1
        if count>=5:
            break
    
    plot_adversarial_examples(examples, confidences)

if __name__=="__main__":
    main()
