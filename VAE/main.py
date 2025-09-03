import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置随机种子以确保可重复性
torch.manual_seed(42)

class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(VAE, self).__init__()
        
        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        # 编码
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        
        # 解码
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    # 重构损失（二元交叉熵）
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # KL散度损失
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + kl_loss, recon_loss, kl_loss

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    recon_losses = 0
    kl_losses = 0
    
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.view(-1, 784).to(device)
        optimizer.zero_grad()
        
        recon_batch, mu, logvar = model(data)
        total_loss, recon_loss, kl_loss = vae_loss(recon_batch, data, mu, logvar)
        
        total_loss.backward()
        train_loss += total_loss.item()
        recon_losses += recon_loss.item()
        kl_losses += kl_loss.item()
        
        optimizer.step()
    
    avg_loss = train_loss / len(train_loader.dataset)
    avg_recon = recon_losses / len(train_loader.dataset)
    avg_kl = kl_losses / len(train_loader.dataset)
    
    return avg_loss, avg_recon, avg_kl

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.view(-1, 784).to(device)
            recon, mu, logvar = model(data)
            total_loss, _, _ = vae_loss(recon, data, mu, logvar)
            test_loss += total_loss.item()
    
    return test_loss / len(test_loader.dataset)

def save_plot(losses, recon_losses, kl_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(losses, label='Total Loss')
    plt.plot(recon_losses, label='Reconstruction Loss')
    plt.plot(kl_losses, label='KL Divergence')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    plt.grid(True)
    plt.savefig('vae_losses.png')
    plt.close()

def visualize_results(model, device, test_loader):
    model.eval()
    with torch.no_grad():
        # 获取一个batch的测试数据
        data, targets = next(iter(test_loader))
        data = data.view(-1, 784).to(device)
        
        # 重建图像
        reconstructions, _, _ = model(data)
        reconstructions = reconstructions.view(-1, 1, 28, 28).cpu().numpy()
        originals = data.view(-1, 1, 28, 28).cpu().numpy()
        
        # 创建4x4的子图
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        fig.subplots_adjust(hspace=0.5)
        
        for i in range(16):
            ax = axes[i // 4, i % 4]
            # 显示原始图像
            ax.imshow(originals[i][0], cmap='gray')
            ax.set_title(f'Label: {targets[i].item()}')
            ax.axis('off')
        
        plt.savefig('original_images.png')
        plt.close()
        
        # 显示重建图像
        fig, axes = plt.subplots(4, 4, figsize=(8, 8))
        fig.subplots_adjust(hspace=0.5)
        
        for i in range(16):
            ax = axes[i // 4, i % 4]
            ax.imshow(reconstructions[i][0], cmap='gray')
            ax.set_title(f'Recon: {targets[i].item()}')
            ax.axis('off')
        
        plt.savefig('reconstructed_images.png')
        plt.close()

def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 超参数
    batch_size = 128
    epochs = 20
    learning_rate = 0.001
    
    # 数据加载
    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 初始化模型
    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练模型
    train_losses = []
    recon_losses = []
    kl_losses = []
    
    for epoch in range(1, epochs + 1):
        avg_loss, avg_recon, avg_kl = train(model, device, train_loader, optimizer, epoch)
        test_loss = test(model, device, test_loader)
        
        train_losses.append(avg_loss)
        recon_losses.append(avg_recon)
        kl_losses.append(avg_kl)
        
        print(f'Epoch {epoch}/{epochs} | Train Loss: {avg_loss:.4f} | Test Loss: {test_loss:.4f}')
    
    # 保存损失曲线
    save_plot(train_losses, recon_losses, kl_losses)
    
    # 保存模型
    torch.save(model.state_dict(), 'vae_mnist.pth')
    print("Model saved to vae_mnist.pth")
    
    # 加载模型
    model = VAE().to(device)
    model.load_state_dict(torch.load('vae_mnist.pth'))
    print("Model loaded from vae_mnist.pth")
    
    # 可视化结果
    visualize_results(model, device, test_loader)
    print("Results saved to original_images.png and reconstructed_images.png")

if __name__ == "__main__":
    main()