import torch
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from test import SimpleMLP  # 替换为你的模型定义文件路径（如main.py）

# -------------------------- 1. 基础配置（含数据加载） --------------------------
# 模型路径（与原训练代码一致）
ORIGINAL_MODEL_PATH = "original_mlp_model.pth"
ATTACKED_MODEL_PATH = "attacked_mlp_model.pth"
# 设备配置（与原训练设备保持一致）
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------- 2. 新增：MNIST测试集加载（复用原代码逻辑） --------------------------
def load_mnist_testset(batch_size=1000):
    """加载MNIST测试集（预处理与原训练代码完全一致，确保准确率公平对比）"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # 原代码的MNIST标准化参数
    ])
    
    test_dataset = datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"成功加载MNIST测试集：共{len(test_dataset)}个样本，批次大小{batch_size}")
    return test_loader

# -------------------------- 3. 新增：模型准确率评估函数 --------------------------
def evaluate_accuracy(model, test_loader, device):
    """评估模型在测试集上的预测准确率"""
    model.eval()  # 切换为评估模式（禁用Dropout、固定BatchNorm等）
    correct = 0  # 正确预测数
    total = 0    # 总样本数
    
    # 评估时不计算梯度，节省内存和时间
    with torch.no_grad():
        for data, targets in test_loader:
            # 数据迁移到目标设备（与模型一致）
            data, targets = data.to(device), targets.to(device)
            
            # 模型推理
            outputs = model(data)
            # 获取预测结果（取输出概率最大的类别）
            _, predicted = torch.max(outputs.data, 1)
            
            # 统计正确数和总数
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    # 计算准确率
    accuracy = 100 * correct / total
    return accuracy

# -------------------------- 4. 原有核心函数（参数加载、比特转换、比特对比） --------------------------
def load_target_model(model_class, model_path, device):
    """加载目标模型（确保参数与设备匹配）"""
    model = model_class().to(device)
    # 加载参数时自动适配设备（避免CPU/GPU不兼容）
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"成功加载模型: {model_path} (设备: {device})")
    return model

def float32_to_bits(x):
    """32位浮点数 → 32位比特列表（无精度丢失）"""
    if isinstance(x, torch.Tensor):
        x = x.cpu().detach().numpy().item()
    elif isinstance(x, np.ndarray):
        x = x.item()
    
    int_repr = np.frombuffer(np.float32(x).tobytes(), dtype=np.uint32)[0]
    return [int(bit) for bit in np.binary_repr(int_repr, width=32)]

def count_bit_differences(param1, param2):
    """逐元素统计两个参数张量的比特差异"""
    if param1.shape != param2.shape:
        raise ValueError(f"参数形状不匹配！param1: {param1.shape}, param2: {param2.shape}")
    
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
    """逐层逐比特对比两个模型的参数"""
    orig_params = dict(original_model.named_parameters())
    att_params = dict(attacked_model.named_parameters())
    
    # 校验参数名称完整性
    orig_names = set(orig_params.keys())
    att_names = set(att_params.keys())
    if orig_names != att_names:
        raise ValueError(
            f"模型参数名称不匹配！\n"
            f"仅原始模型有: {orig_names - att_names}\n"
            f"仅攻击后模型有: {att_names - orig_names}"
        )
    
    # 逐层统计比特差异
    comparison_result = defaultdict(dict)
    global_total_diff = 0
    global_total_bits = 0
    
    for param_name in sorted(orig_names):
        orig_param = orig_params[param_name]
        att_param = att_params[param_name]
        
        elem_diff_list, param_total_diff, param_total_elem = count_bit_differences(orig_param, att_param)
        param_total_bits = param_total_elem * 32
        
        comparison_result[param_name] = {
            "param_shape": orig_param.shape,
            "total_elements": param_total_elem,
            "total_bits": param_total_bits,
            "total_bit_diff": param_total_diff,
            "bit_diff_rate": param_total_diff / param_total_bits if param_total_bits > 0 else 0.0,
            "element_bit_diff_sample": elem_diff_list[:5]
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

# -------------------------- 5. 新增：整合准确率对比与结果打印 --------------------------
def print_complete_comparison(acc_original, acc_attacked, bitwise_result):
    """打印“准确率对比 + 参数比特对比”的完整报告"""
    print("=" * 90)
    print("          模型完整对比报告 (MNIST测试集)")
    print("=" * 90)
    
    # 1. 先打印准确率对比（核心新增部分）
    print("\n【1. 预测准确率对比】")
    print("-" * 50)
    print(f"原始模型准确率: {acc_original:.4f}%")
    print(f"攻击后模型准确率: {acc_attacked:.4f}%")
    print(f"准确率下降幅度: {acc_original - acc_attacked:.4f}%")
    # 补充准确率差异等级提示
    drop = acc_original - acc_attacked
    if drop < 1.0:
        print(f"→ 差异等级: 轻微（准确率下降<1%）")
    elif 1.0 <= drop < 10.0:
        print(f"→ 差异等级: 中等（准确率下降1%-10%）")
    else:
        print(f"→ 差异等级: 显著（准确率下降≥10%）")
    
    # 2. 再打印参数比特对比（原有逻辑）
    print("\n【2. 参数逐比特对比】")
    print("-" * 50)
    param_names = [name for name in bitwise_result if name != "GLOBAL_SUMMARY"]
    for param_name in param_names:
        res = bitwise_result[param_name]
        print(f"\n【参数层】: {param_name}")
        print(f"  - 形状: {res['param_shape']} | 总元素数: {res['total_elements']:,}")
        print(f"  - 总比特数: {res['total_bits']:,} | 比特差异数: {res['total_bit_diff']:,}")
        print(f"  - 比特差异率: {res['bit_diff_rate']:.6f} (≈ {res['bit_diff_rate']*100:.4f}%)")
        print(f"  - 前5元素比特差异: {res['element_bit_diff_sample']}")
    
    # 3. 全局汇总
    global_res = bitwise_result["GLOBAL_SUMMARY"]
    print("\n【3. 全局汇总】")
    print("-" * 50)
    print(f"模型总比特数: {global_res['total_bits_across_model']:,} (≈ {global_res['total_bits_across_model']/1e6:.1f}M)")
    print(f"模型总比特差异数: {global_res['total_bit_diff_across_model']:,}")
    print(f"模型全局比特差异率: {global_res['global_bit_diff_rate']:.6f} (≈ {global_res['global_bit_diff_rate']*100:.4f}%)")
    print("=" * 90)

# -------------------------- 6. 主执行逻辑（整合所有功能） --------------------------
if __name__ == "__main__":
    try:
        # 步骤1: 加载MNIST测试集
        test_loader = load_mnist_testset(batch_size=1000)
        
        # 步骤2: 加载两个模型
        original_model = load_target_model(SimpleMLP, ORIGINAL_MODEL_PATH, DEVICE)
        attacked_model = load_target_model(SimpleMLP, ATTACKED_MODEL_PATH, DEVICE)
        
        # 步骤3: 评估两个模型的准确率（核心新增步骤）
        print("\n开始评估模型准确率...")
        acc_original = evaluate_accuracy(original_model, test_loader, DEVICE)
        acc_attacked = evaluate_accuracy(attacked_model, test_loader, DEVICE)
        
        # 步骤4: 逐比特对比模型参数
        print("\n开始逐比特对比模型参数...")
        bitwise_result = compare_models_bitwise(original_model, attacked_model)
        
        # 步骤5: 打印完整对比报告
        print("\n" + "="*20 + " 对比完成，生成报告 " + "="*20)
        print_complete_comparison(acc_original, acc_attacked, bitwise_result)
    
    except Exception as e:
        print(f"\n对比过程出错: {str(e)}")
        print("请检查：1. 模型路径 2. 模型定义一致性 3. 数据下载权限 4. 设备兼容性")