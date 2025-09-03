import torch
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import SimpleCNN  # 替换为你的CNN模型定义文件路径

# -------------------------- 1. 基础配置 --------------------------
# 模型路径（与CNN训练代码一致）
ORIGINAL_MODEL_PATH = "original_model.pth"    # CNN原始模型
ATTACKED_MODEL_PATH = "attacked_model.pth"    # CNN攻击后模型
# 设备配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------- 2. 数据加载（MNIST测试集） --------------------------
def load_mnist_testset(batch_size=1000):
    """加载MNIST测试集（与训练时预处理一致）"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # 保持与训练代码相同的标准化参数
    ])
    
    test_dataset = datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"已加载MNIST测试集：{len(test_dataset)}个样本")
    return test_loader

# -------------------------- 3. 模型准确率评估 --------------------------
def evaluate_accuracy(model, test_loader, device):
    """评估模型在测试集上的准确率（兼容CNN结构）"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)  # CNN的forward会自动处理28x28图像输入
            _, predicted = torch.max(outputs.data, 1)
            
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

# -------------------------- 4. 模型加载与参数比特对比 --------------------------
def load_target_model(model_class, model_path, device):
    """加载CNN模型（兼容卷积层参数结构）"""
    model = model_class().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"已加载模型: {model_path} (设备: {device})")
    return model

def float32_to_bits(x):
    """将32位浮点数转换为32位比特列表（核心工具函数）"""
    if isinstance(x, torch.Tensor):
        x = x.cpu().detach().numpy().item()
    elif isinstance(x, np.ndarray):
        x = x.item()
    
    # 浮点数→字节→无符号整数→比特列表
    int_repr = np.frombuffer(np.float32(x).tobytes(), dtype=np.uint32)[0]
    return [int(bit) for bit in np.binary_repr(int_repr, width=32)]

def count_bit_differences(param1, param2):
    """逐元素统计两个参数张量的比特差异（支持CNN的4维卷积参数）"""
    if param1.shape != param2.shape:
        raise ValueError(f"参数形状不匹配！param1: {param1.shape}, param2: {param2.shape}")
    
    # 展平任意维度的参数（卷积层为4维，全连接层为2维）
    param1_flat = param1.cpu().detach().flatten()
    param2_flat = param2.cpu().detach().flatten()
    total_elements = param1_flat.numel()
    
    element_bit_diff = []
    total_bit_diff = 0
    for val1, val2 in zip(param1_flat, param2_flat):
        bits1 = float32_to_bits(val1)
        bits2 = float32_to_bits(val2)
        diff_count = sum(b1 != b2 for b1, b2 in zip(bits1, bits2))
        element_bit_diff.append(diff_count)
        total_bit_diff += diff_count
    
    return element_bit_diff, total_bit_diff, total_elements

def compare_models_bitwise(original_model, attacked_model):
    """对比两个CNN模型的所有参数（卷积层+全连接层）"""
    orig_params = dict(original_model.named_parameters())
    att_params = dict(attacked_model.named_parameters())
    
    # 校验参数名称完整性（CNN包含conv1和fc相关参数）
    orig_names = set(orig_params.keys())
    att_names = set(att_params.keys())
    if orig_names != att_names:
        raise ValueError(
            f"模型参数名称不匹配！\n"
            f"仅原始模型有: {orig_names - att_names}\n"
            f"仅攻击后模型有: {att_names - orig_names}"
        )
    
    # 逐层对比（卷积层参数为4维，全连接层为2维）
    comparison_result = defaultdict(dict)
    global_total_diff = 0
    global_total_bits = 0
    
    for param_name in sorted(orig_names):
        orig_param = orig_params[param_name]
        att_param = att_params[param_name]
        
        elem_diff_list, param_total_diff, param_total_elem = count_bit_differences(orig_param, att_param)
        param_total_bits = param_total_elem * 32  # 每个元素32比特
        
        comparison_result[param_name] = {
            "param_shape": orig_param.shape,          # 卷积层为[out_channels, in_channels, kH, kW]
            "total_elements": param_total_elem,
            "total_bits": param_total_bits,
            "total_bit_diff": param_total_diff,
            "bit_diff_rate": param_total_diff / param_total_bits if param_total_bits > 0 else 0.0,
            "element_bit_diff_sample": elem_diff_list[:5]  # 前5个元素的比特差异示例
        }
        
        global_total_diff += param_total_diff
        global_total_bits += param_total_bits
    
    # 全局汇总
    comparison_result["GLOBAL_SUMMARY"] = {
        "total_bits_across_model": global_total_bits,
        "total_bit_diff_across_model": global_total_diff,
        "global_bit_diff_rate": global_total_diff / global_total_bits if global_total_bits > 0 else 0.0
    }
    
    return comparison_result

# -------------------------- 5. 完整对比报告打印 --------------------------
def print_complete_comparison(acc_original, acc_attacked, bitwise_result):
    """打印CNN模型的准确率对比和参数比特对比报告"""
    print("=" * 90)
    print("          CNN模型完整对比报告 (原始模型 vs 攻击后模型)")
    print("=" * 90)
    
    # 1. 准确率对比
    print("\n【1. 测试集准确率对比】")
    print("-" * 50)
    print(f"原始CNN模型准确率: {acc_original:.4f}%")
    print(f"攻击后CNN模型准确率: {acc_attacked:.4f}%")
    print(f"准确率下降幅度: {acc_original - acc_attacked:.4f}%")
    
    # 准确率差异等级
    drop = acc_original - acc_attacked
    if drop < 1.0:
        print(f"→ 差异等级: 轻微（攻击影响小）")
    elif 1.0 <= drop < 10.0:
        print(f"→ 差异等级: 中等（攻击有明显影响）")
    else:
        print(f"→ 差异等级: 显著（攻击严重破坏模型性能）")
    
    # 2. 参数比特对比（重点展示卷积层和全连接层）
    print("\n【2. 参数逐比特对比】")
    print("-" * 50)
    param_names = [name for name in bitwise_result if name != "GLOBAL_SUMMARY"]
    for param_name in param_names:
        res = bitwise_result[param_name]
        print(f"\n【参数层】: {param_name}")
        print(f"  - 形状: {res['param_shape']} (卷积层为[out, in, kH, kW]，全连接层为[out, in])")
        print(f"  - 总元素数: {res['total_elements']:,} | 总比特数: {res['total_bits']:,}")
        print(f"  - 比特差异数: {res['total_bit_diff']:,} | 差异率: {res['bit_diff_rate']*100:.4f}%")
        print(f"  - 前5元素比特差异: {res['element_bit_diff_sample']}")
    
    # 3. 全局汇总
    global_res = bitwise_result["GLOBAL_SUMMARY"]
    print("\n【3. 全局汇总】")
    print("-" * 50)
    print(f"模型总比特数: {global_res['total_bits_across_model']:,} (≈ {global_res['total_bits_across_model']/1e6:.1f}M)")
    print(f"总比特差异数: {global_res['total_bit_diff_across_model']:,}")
    print(f"全局比特差异率: {global_res['global_bit_diff_rate']*100:.4f}%")
    print("=" * 90)

# -------------------------- 6. 主执行逻辑 --------------------------
if __name__ == "__main__":
    try:
        # 步骤1: 加载测试集
        test_loader = load_mnist_testset(batch_size=1000)
        
        # 步骤2: 加载两个CNN模型
        original_model = load_target_model(SimpleCNN, ORIGINAL_MODEL_PATH, DEVICE)
        attacked_model = load_target_model(SimpleCNN, ATTACKED_MODEL_PATH, DEVICE)
        
        # 步骤3: 评估准确率
        print("\n开始评估模型准确率...")
        acc_original = evaluate_accuracy(original_model, test_loader, DEVICE)
        acc_attacked = evaluate_accuracy(attacked_model, test_loader, DEVICE)
        
        # 步骤4: 逐比特对比参数
        print("\n开始对比模型参数（支持卷积层4维参数）...")
        bitwise_result = compare_models_bitwise(original_model, attacked_model)
        
        # 步骤5: 打印完整报告
        print("\n" + "="*20 + " 对比完成 " + "="*20)
        print_complete_comparison(acc_original, acc_attacked, bitwise_result)
    
    except Exception as e:
        print(f"\n执行出错: {str(e)}")
        print("请检查：1. 模型路径是否正确 2. SimpleCNN定义是否一致 3. 数据是否下载成功")
