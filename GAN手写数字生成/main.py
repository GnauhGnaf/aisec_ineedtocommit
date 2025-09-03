import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from torchvision.utils import save_image, make_grid
import matplotlib as mpl

# 设置中文字体支持
try:
    # 尝试使用系统支持的中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'WenQuanYi Micro Hei', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
except:
    # 如果无法设置中文字体，使用英文替代
    print("警告: 无法设置中文字体，将使用英文显示")

# 设置随机种子以确保可重现性
torch.manual_seed(42)
np.random.seed(42)

# 检查GPU是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 创建保存生成图像的目录
os.makedirs("dcgan_images", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)

# 定义超参数
latent_dim = 100  # 噪声向量维度
batch_size = 128
epochs = 300
lr = 0.0002
beta1 = 0.5  # Adam优化器的参数

# 数据预处理 - 将图像归一化到[-1, 1]范围
transform = transforms.Compose([
    transforms.Resize(64),  # 放大图像以获得更好的质量
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # 将像素值从[0,1]映射到[-1,1]
])

# 加载MNIST数据集
train_dataset = datasets.MNIST(
    root='./data', 
    train=True,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size,
    shuffle=True
)

# 定义生成器网络 (DCGAN架构)
class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            # 输入: latent_dim维噪声
            nn.ConvTranspose2d(latent_dim, 512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            # 输出尺寸: (512, 4, 4)
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            # 输出尺寸: (256, 8, 8)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            # 输出尺寸: (128, 16, 16)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # 输出尺寸: (64, 32, 32)
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
            # 输出尺寸: (1, 64, 64)
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, z):
        # 将噪声向量重塑为4D张量 (batch_size, latent_dim, 1, 1)
        z = z.view(z.size(0), z.size(1), 1, 1)
        return self.model(z)

# 定义判别器网络 (DCGAN架构)
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            # 输入: 1 x 64 x 64
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 输出尺寸: (64, 32, 32)
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 输出尺寸: (128, 16, 16)
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 输出尺寸: (256, 8, 8)
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 输出尺寸: (512, 4, 4)
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Flatten(),  # 添加Flatten层解决形状问题
            nn.Sigmoid()
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, img):
        return self.model(img)

# 初始化生成器和判别器
generator = Generator(latent_dim).to(device)
discriminator = Discriminator().to(device)

# 定义损失函数和优化器
adversarial_loss = nn.BCELoss().to(device)

optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
scheduler_G = torch.optim.lr_scheduler.ExponentialLR(optimizer_G, gamma=0.98)
scheduler_D = torch.optim.lr_scheduler.ExponentialLR(optimizer_D, gamma=0.98)
# 用于可视化的固定噪声向量
fixed_noise = torch.randn(16, latent_dim, device=device)

# 训练GAN
G_losses = []
D_losses = []

# 创建实时更新的损失曲线图表（但不显示）
plt.figure(figsize=(10, 6))
plt.title('DCGAN Training Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.grid(True, linestyle='--', alpha=0.6)
line_g, = plt.plot([], [], label='Generator Loss', color='blue', alpha=0.7)
line_d, = plt.plot([], [], label='Discriminator Loss', color='red', alpha=0.7)
plt.legend()
plt.tight_layout()
plt.savefig("dcgan_loss_curve.png", dpi=150)
plt.close()  # 关闭图表避免内存泄漏
print("初始化损失曲线图已创建: dcgan_loss_curve.png")

print("开始训练DCGAN...")
for epoch in range(epochs):
    for i, (imgs, _) in enumerate(train_loader):
        
        # 配置输入
        real_imgs = imgs.to(device)
        current_batch_size = real_imgs.size(0)  # 使用当前批次大小，避免最后一批大小不一致
        
        # 真实和假的标签
        real_labels = torch.full((current_batch_size, 1), 0.9, device=device)  # 不是精确的1.0
        fake_labels = torch.full((current_batch_size, 1), 0.1, device=device)  # 不是精确的0.0
        
        # =====================
        #  训练判别器
        # =====================
        
        optimizer_D.zero_grad()
        
        # 真实图像的损失
        real_output = discriminator(real_imgs)
        d_loss_real = adversarial_loss(real_output, real_labels)
        
        # 假图像的损失
        z = torch.randn(current_batch_size, latent_dim, device=device)
        fake_imgs = generator(z)
        fake_output = discriminator(fake_imgs.detach())
        d_loss_fake = adversarial_loss(fake_output, fake_labels)
        
        # 总判别器损失
        d_loss = d_loss_real + d_loss_fake
        d_loss.backward()
        optimizer_D.step()
        
        # =====================
        #  训练生成器
        # =====================
        
        optimizer_G.zero_grad()
        
        # 生成一批图像
        gen_imgs = generator(z)
        
        # 生成器希望判别器将假图像分类为真实
        g_output = discriminator(gen_imgs)
        g_loss = adversarial_loss(g_output, real_labels)
        
        g_loss.backward()
        optimizer_G.step()
        
        # 记录损失
        if i % 50 == 0:
            print(f"[Epoch {epoch+1}/{epochs}] [Batch {i}/{len(train_loader)}] "
                  f"[D loss: {d_loss.item():.6f}] [G loss: {g_loss.item():.6f}]")
            sys.stdout.flush()  # 刷新输出缓冲区
        
        G_losses.append(g_loss.item())
        D_losses.append(d_loss.item())
        
        # 每50步更新一次损失曲线
        if len(G_losses) % 50 == 0:
            # 只绘制最近1000个点以保持图表清晰
            window_size = min(1000, len(G_losses))
            x = np.arange(len(G_losses) - window_size, len(G_losses))
            
            # 创建新图表进行更新
            plt.figure(figsize=(10, 6))
            plt.title('DCGAN Training Loss')
            plt.xlabel('Iterations')
            plt.ylabel('Loss')
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.plot(x, G_losses[-window_size:], label='Generator Loss', color='blue', alpha=0.7)
            plt.plot(x, D_losses[-window_size:], label='Discriminator Loss', color='red', alpha=0.7)
            plt.legend()
            plt.xlim(len(G_losses) - window_size, len(G_losses))
            plt.ylim(0, max(1, max(G_losses[-window_size:] + D_losses[-window_size:]) * 1.1))
            plt.tight_layout()
            
            # 保存更新的图表
            plt.savefig("dcgan_loss_curve.png", dpi=150)
            plt.close()  # 关闭图表避免内存泄漏
        # 每1000个点保存一次完整损失曲线
        if len(G_losses) % 1000 == 0 and len(G_losses) > 0:
            plt.figure(figsize=(12, 8))
            plt.title('Full DCGAN Training Loss')
            plt.xlabel('Iterations')
            plt.ylabel('Loss')
            plt.grid(True, linestyle='--', alpha=0.6)
            
            # 绘制完整损失曲线
            plt.plot(G_losses, label='Generator Loss', color='blue', alpha=0.7)
            plt.plot(D_losses, label='Discriminator Loss', color='red', alpha=0.7)
            
            # 添加平滑曲线
            def moving_average(data, window_size=100):
                return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
            
            smoothed_G = moving_average(G_losses)
            smoothed_D = moving_average(D_losses)
            x_smoothed = np.arange(len(smoothed_G)) + 50  # 居中显示
            
            plt.plot(x_smoothed, smoothed_G, label='Generator (Smoothed)', color='darkblue', linestyle='-', linewidth=2)
            plt.plot(x_smoothed, smoothed_D, label='Discriminator (Smoothed)', color='darkred', linestyle='-', linewidth=2)
            
            # 标记当前窗口位置
            if len(G_losses) > 1000:
                plt.axvline(x=len(G_losses)-1000, color='gray', linestyle='--', alpha=0.5)
                plt.text(len(G_losses)-1000, max(plt.ylim()[0], 0.1), 
                        f'Recent 1k window\n(iter {len(G_losses)-1000}-{len(G_losses)})',
                        fontsize=9, ha='right', va='bottom', bbox=dict(facecolor='white', alpha=0.8))
            
            plt.legend()
            plt.tight_layout()
            
            # 保存完整损失曲线
            plt.savefig(f"dcgan_full_loss_curve_{len(G_losses):06d}.png", dpi=200)
            plt.close()
            print(f"保存完整损失曲线图: dcgan_full_loss_curve_{len(G_losses):06d}.png")
    if epoch > 50:  # 50个epoch后开始衰减
        scheduler_G.step()
        scheduler_D.step()
    # 每个epoch结束后保存生成的图像
    with torch.no_grad():
        fake_imgs = generator(fixed_noise).detach().cpu()
        save_image(fake_imgs, f"dcgan_images/epoch_{epoch+1:03d}.png", nrow=4, normalize=True)
      
    # 每10个epoch保存一次模型
    if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
        torch.save(generator.state_dict(), f"saved_models/generator_epoch_{epoch+1}.pth")
        torch.save(discriminator.state_dict(), f"saved_models/discriminator_epoch_{epoch+1}.pth")
        print(f"模型在epoch {epoch+1}保存成功")

# ==================================================================
# 训练完成后绘制最终损失曲线
# ==================================================================
print("训练完成! 绘制最终损失曲线...")

# 创建更高质量的最终损失曲线图
plt.figure(figsize=(12, 6))

# 绘制完整损失曲线
plt.plot(G_losses, label='Generator Loss', color='blue', alpha=0.7)
plt.plot(D_losses, label='Discriminator Loss', color='red', alpha=0.7)

# 添加平滑曲线（移动平均）
def moving_average(data, window_size=100):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

smoothed_G = moving_average(G_losses, window_size=50)
smoothed_D = moving_average(D_losses, window_size=50)

# 调整平滑曲线的x轴位置
x_smoothed = np.arange(len(smoothed_G)) + 25  # 居中显示

plt.plot(x_smoothed, smoothed_G, label='Generator (Smoothed)', color='darkblue', linestyle='--', linewidth=2)
plt.plot(x_smoothed, smoothed_D, label='Discriminator (Smoothed)', color='darkred', linestyle='--', linewidth=2)

# 设置图表属性
plt.title('Loss Plot - Generator vs Discriminator', fontsize=14)
plt.xlabel('Iterations', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)

# 添加说明文字（使用英文避免字体问题）
plt.figtext(0.5, 0.01, 
            f"DCGAN Training on MNIST | Epochs: {epochs} | Batch Size: {batch_size} | LR: {lr}",
            ha="center", fontsize=10, bbox={"facecolor":"lightgray", "alpha":0.5, "pad":5})

plt.tight_layout()

# 保存最终图表
final_loss_plot_path = "dcgan_final_loss_curve.png"
plt.savefig(final_loss_plot_path, dpi=300, bbox_inches='tight')
print(f"最终损失曲线图已保存至: {final_loss_plot_path}")

# 保存损失数据
np.save("G_losses.npy", np.array(G_losses))
np.save("D_losses.npy", np.array(D_losses))
print("损失数据已保存为G_losses.npy和D_losses.npy")

# 显示图表（仅在支持图形界面的环境中）
if 'DISPLAY' in os.environ or sys.platform == 'win32':
    try:
        plt.show()
    except:
        print("无法显示图表，请查看保存的图像文件")
else:
    print("当前环境无图形界面支持，请查看保存的图像文件")

# ==================================================================
# 继续原有代码
# ==================================================================
print("\n===== 训练完成! =====")
print(f"最终模型保存在: saved_models/generator_epoch_{epochs}.pth")

# 确认最终图像文件是否存在
final_img_path = f"dcgan_images/epoch_{epochs:03d}.png"
if os.path.exists(final_img_path):
    print(f"最终生成的图像已保存到: {final_img_path}")
    
    # 尝试显示图像（仅在支持GUI的环境中）
    if 'DISPLAY' in os.environ or sys.platform == 'win32':
        try:
            plt.figure(figsize=(10, 10))
            final_img = plt.imread(final_img_path)
            plt.imshow(final_img)
            plt.axis('off')
            # 使用英文标题避免字体问题
            plt.title(f"Final Generated MNIST Images (Epoch {epochs})", fontsize=14)
            plt.tight_layout()
            plt.show(block=True)  # 确保窗口保持打开
        except Exception as e:
            print(f"无法显示图像: {e}")
    else:
        print("当前环境无图形界面支持，请查看保存的图像文件")
else:
    print(f"警告: 最终图像文件未找到 - {final_img_path}")

# 确保所有图像都被正确保存
print("\n保存的重要文件:")
print(f"1. 实时损失曲线: dcgan_loss_curve.png")
print(f"2. 最终损失曲线: {final_loss_plot_path}")
print(f"3. 最终图像: {final_img_path}")
print(f"4. 模型文件: saved_models/ 目录")

# 生成并保存最终样本图像
print("\n生成最终样本图像...")
with torch.no_grad():
    # 生成64个图像
    z = torch.randn(64, latent_dim, device=device)
    gen_imgs = generator(z).detach().cpu()

# 显示生成的图像网格
plt.figure(figsize=(12, 12))
plt.axis("off")
# 使用英文标题避免字体问题
plt.title("Generated MNIST Handwritten Digits", fontsize=16)
grid_img = make_grid(gen_imgs, nrow=8, padding=2, normalize=True)
plt.imshow(np.transpose(grid_img.numpy(), (1, 2, 0)))
plt.savefig("dcgan_mnist_grid.png", dpi=150, bbox_inches='tight')
print("图像网格已保存: dcgan_mnist_grid.png")

# 尝试创建训练过程动画
try:
    # 使用新版本imageio以避免警告
    import imageio.v2 as imageio
    print("创建训练过程动画...")
    images = []
    for epoch in range(epochs):
        img_path = f"dcgan_images/epoch_{epoch+1:03d}.png"
        if os.path.exists(img_path):
            images.append(imageio.imread(img_path))
    imageio.mimsave('dcgan_training.gif', images, fps=5)
    print("已创建训练过程动画: dcgan_training.gif")
    
except ImportError:
    print("要创建训练过程动画，请安装imageio: pip install imageio")

# 额外保存一个最终模型的样本
with torch.no_grad():
    z = torch.randn(64, latent_dim, device=device)
    gen_imgs = generator(z).detach().cpu()
    save_image(gen_imgs, "dcgan_final_samples.png", nrow=8, normalize=True)
    print("额外保存的样本: dcgan_final_samples.png")

print("\n所有任务已完成!")