import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import copy
import random
import os
from collections import defaultdict

# 设置随机种子以确保结果可复现
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# 模型保存路径
ORIGINAL_MODEL_PATH = "original_mlp_model.pth"
ATTACKED_MODEL_PATH = "attacked_mlp_model.pth"

# 1. 定义MLP模型（替换CNN）
class SimpleMLP(nn.Module):
    def __init__(self, input_size=784, hidden_sizes=[256, 128], output_size=10):
        super(SimpleMLP, self).__init__()
        # 定义MLP结构
        layers = []
        # 输入层到第一个隐藏层
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())
        # 添加其他隐藏层
        for i in range(1, len(hidden_sizes)):
            layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            layers.append(nn.ReLU())
        # 输出层
        layers.append(nn.Linear(hidden_sizes[-1], output_size))
        
        # 将所有层组合成序列
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        # MLP需要平展输入（28x28 -> 784）
        x = x.view(x.size(0), -1)  # 平展操作
        return self.model(x)

# 2. 数据加载和预处理
def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    return train_loader, test_loader

# 3. 模型训练函数
def train_model(model, train_loader, device, epochs=20):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if batch_idx % 100 == 99:
                print(f'Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {running_loss/100:.4f}')
                running_loss = 0.0
    print("训练完成")
    return model

# 4. 模型保存和加载函数
def save_model(model, path):
    """保存模型到指定路径"""
    torch.save(model.state_dict(), path)
    print(f"模型已保存到 {path}")

def load_model(model_class, path, device):
    """从指定路径加载模型"""
    model = model_class().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()  # 设置为评估模式
    print(f"已从 {path} 加载模型")
    return model

# 5. 模型评估函数
def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    print(f'模型准确率: {accuracy:.2f}%')
    return accuracy

# 6. 提取模型所有参数的函数
def extract_model_parameters(model):
    """提取模型所有参数并以字典形式返回"""
    params = {}
    for name, param in model.named_parameters():
        # 保存参数的副本，避免引用问题
        params[name] = (param.data.cpu().numpy().copy(), param.shape)
    return params

# 7. 对比两个模型参数的函数
def compare_parameters(original_params, attacked_params):
    """对比两个模型的所有参数，返回差异信息"""
    comparison = defaultdict(dict)
    
    # 首先检查所有参数名称是否相同
    original_names = set(original_params.keys())
    attacked_names = set(attacked_params.keys())
    
    if original_names != attacked_names:
        comparison['mismatched_params'] = {
            'original_only': original_names - attacked_names,
            'attacked_only': attacked_names - original_names
        }
    
    # 对比每个参数
    for name in original_names & attacked_names:
        orig_data, orig_shape = original_params[name]
        att_data, att_shape = attacked_params[name]
        
        # 检查形状是否相同
        if orig_shape != att_shape:
            comparison[name]['shape_mismatch'] = {
                'original': orig_shape,
                'attacked': att_shape
            }
            continue
        
        # 检查是否有差异
        if np.array_equal(orig_data, att_data):
            comparison[name]['status'] = 'unchanged'
            comparison[name]['elements'] = f"{orig_data.size} elements (all unchanged)"
        else:
            # 找到所有不同的元素
            diff_mask = ~np.isclose(orig_data, att_data)
            diff_count = np.sum(diff_mask)
            total_count = orig_data.size
            
            comparison[name]['status'] = 'changed'
            comparison[name]['elements'] = f"{diff_count}/{total_count} elements changed"
            comparison[name]['differences'] = []
            
            # 收集差异点（对于大型参数，我们限制显示数量）
            max_display = 10  # 最多显示10个差异点
            diff_indices = np.where(diff_mask)
            
            for i in range(min(diff_count, max_display)):
                indices = tuple(idx[i] for idx in diff_indices)
                orig_val = orig_data[indices]
                att_val = att_data[indices]
                
                # 计算比特差异
                orig_bits = float_to_bitstring(orig_val)
                att_bits = float_to_bitstring(att_val)
                bit_diff_pos = [i for i, (b1, b2) in enumerate(zip(orig_bits, att_bits)) if b1 != b2]
                
                comparison[name]['differences'].append({
                    'indices': indices,
                    'original_value': orig_val,
                    'attacked_value': att_val,
                    'original_bits': orig_bits,
                    'attacked_bits': att_bits,
                    'bit_differences': bit_diff_pos
                })
            
            if diff_count > max_display:
                comparison[name]['differences'].append({
                    'note': f"... and {diff_count - max_display} more differences"
                })
    
    return comparison

# 辅助函数：将浮点数转换为32位比特字符串
def float_to_bitstring(f):
    if np.isnan(f):
        return "NaN"
    # 将浮点数转换为32位无符号整数表示
    int_repr = np.frombuffer(np.float32(f).tobytes(), dtype=np.uint32)[0]
    # 转换为32位二进制字符串
    return np.binary_repr(int_repr, width=32)

# 8. 打印参数对比结果的函数
def print_parameter_comparison(comparison):
    """打印参数对比结果"""
    print("\n===== 模型参数对比 (原始 vs 攻击后) =====")
    
    # 检查参数名称是否匹配
    if 'mismatched_params' in comparison:
        print("\n参数名称不匹配:")
        if comparison['mismatched_params']['original_only']:
            print(f"  仅原始模型有: {', '.join(comparison['mismatched_params']['original_only'])}")
        if comparison['mismatched_params']['attacked_only']:
            print(f"  仅攻击后模型有: {', '.join(comparison['mismatched_params']['attacked_only'])}")
    
    # 打印每个参数的对比结果
    param_names = [name for name in comparison if name != 'mismatched_params']
    for name in sorted(param_names):
        print(f"\n参数: {name}")
        param_info = comparison[name]
        
        if 'shape_mismatch' in param_info:
            print(f"  形状不匹配 - 原始: {param_info['shape_mismatch']['original']}, 攻击后: {param_info['shape_mismatch']['attacked']}")
            continue
        
        print(f"  状态: {param_info['status']}")
        print(f"  元素: {param_info['elements']}")
        
        if param_info['status'] == 'changed' and 'differences' in param_info:
            print("  差异点:")
            for i, diff in enumerate(param_info['differences']):
                if 'note' in diff:
                    print(f"    {diff['note']}")
                    continue
                print(f"    差异 #{i+1}:")
                print(f"      位置: {diff['indices']}")
                print(f"      原始值: {diff['original_value']:.6f}")
                print(f"      攻击后值: {diff['attacked_value']:.6f}")
                print(f"      比特差异位置: {diff['bit_differences']}")

# 9. 比特翻转攻击核心函数
class BitFlipAttacker:
    def __init__(self, model, device):
        self.model = copy.deepcopy(model)
        self.device = device
        # 存储翻转操作的记录，便于后续验证
        self.flip_records = []
        
    def float_to_bits(self, x):
        """将32位浮点数转换为比特列表"""
        if isinstance(x, torch.Tensor):
            x = x.cpu().detach().numpy()
        # 将浮点数转换为32位无符号整数表示
        int_repr = np.frombuffer(np.float32(x).tobytes(), dtype=np.uint32)[0]
        # 转换为32位比特列表
        return [int(bit) for bit in np.binary_repr(int_repr, width=32)]
    
    def bits_to_float(self, bits):
        """将比特列表转换回32位浮点数（返回PyTorch张量）"""
        # 将比特列表转换为整数
        bit_string = ''.join(str(bit) for bit in bits)
        int_repr = np.uint32(int(bit_string, 2))
        # 转换回浮点数并转为PyTorch张量
        return torch.tensor(int_repr.view(np.float32), dtype=torch.float32, device=self.device)
    
    def flip_bit(self, param, bit_position, param_name):
        """仅翻转参数中一个随机元素的指定比特位（每次只修改1个比特）"""
        # 保存原始参数形状
        original_shape = param.shape
        # 展平参数以便随机选择单个元素
        param_flat = param.flatten()
        
        # 随机选择1个参数元素的索引
        num_elements = param_flat.numel()
        target_idx = torch.randint(0, num_elements, (1,)).item()
        
        # 保存翻转记录
        original_value = param_flat[target_idx].cpu().item()
        self.flip_records.append({
            'param_name': param_name,
            'element_index': target_idx,
            'original_shape': original_shape,
            'bit_position': bit_position,
            'original_value': original_value
        })
        
        # 仅对目标元素进行比特翻转
        target_bits = self.float_to_bits(param_flat[target_idx])
        target_bits[bit_position] = 1 - target_bits[bit_position]
        param_flat[target_idx] = self.bits_to_float(target_bits)
        
        # 恢复原始形状并确保张量在正确设备上
        return param_flat.reshape(original_shape).to(self.device)
    
    def find_critical_bits(self, test_loader, top_k=3):
        """找到对模型性能影响最大的k个比特"""
        critical_bits = []
        original_accuracy = evaluate_model(self.model, test_loader, self.device)
        print(f"原始准确率: {original_accuracy:.2f}%")
        
        # 遍历模型所有参数
        for name, param in self.model.named_parameters():
            original_param = copy.deepcopy(param.data)
            
            # 测试每个比特位置 (0-31)
            for bit_pos in range(32):
                # 为了加快演示，我们只测试部分比特位
                if bit_pos > 10:  # 只测试前11位
                    break
                    
                modified_param = self.flip_bit(original_param, bit_pos, name)
                param.data = modified_param
                
                acc = evaluate_model(self.model, test_loader, self.device)
                accuracy_drop = original_accuracy - acc
                
                critical_bits.append({
                    'param_name': name,
                    'bit_position': bit_pos,
                    'accuracy_drop': accuracy_drop,
                    'accuracy_after_attack': acc
                })
                
                # 恢复原始参数
                param.data = original_param
            
            # 为了演示速度，只处理前两个层
            if "2.weight" in name:  # 对于MLP，我们检查第二层的权重
                break
        
        # 按准确率下降程度排序
        critical_bits.sort(key=lambda x: x['accuracy_drop'], reverse=True)
        return critical_bits[:top_k]
    
    def perform_attack(self, test_loader, critical_bits):
        """使用找到的关键比特进行攻击"""
        attacked_model = copy.deepcopy(self.model)
        self.flip_records = []  # 重置翻转记录
        
        # 应用所有关键比特翻转
        for cb in critical_bits:
            for name, param in attacked_model.named_parameters():
                if name == cb['param_name']:
                    param.data = self.flip_bit(param.data, cb['bit_position'], name)
                    break
        
        # 评估攻击后的模型
        final_accuracy = evaluate_model(attacked_model, test_loader, self.device)
        return attacked_model, final_accuracy, self.flip_records

# 10. 主函数
def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载数据
    train_loader, test_loader = load_data()
    
    # 检查是否存在已保存的原始模型
    if os.path.exists(ORIGINAL_MODEL_PATH):
        # 加载已保存的模型
        model = load_model(SimpleMLP, ORIGINAL_MODEL_PATH, device)
    else:
        # 初始化并训练新模型
        print("未找到已保存的模型，开始训练新模型...")
        model = SimpleMLP().to(device)
        model = train_model(model, train_loader, device, epochs=20)
        # 保存训练好的模型
        save_model(model, ORIGINAL_MODEL_PATH)
    
    # 保存原始模型参数用于对比
    original_model = copy.deepcopy(model)
    original_params = extract_model_parameters(original_model)
    
    # 评估原始模型
    original_accuracy = evaluate_model(model, test_loader, device)
    
    # 初始化攻击者
    attacker = BitFlipAttacker(model, device)
    
    # 找到关键比特
    print("\n正在寻找关键比特...")
    critical_bits = attacker.find_critical_bits(test_loader, top_k=3)
    
    # 显示关键比特信息
    print("\n发现的关键比特:")
    for i, cb in enumerate(critical_bits):
        print(f"关键比特 #{i+1}:")
        print(f"  参数名称: {cb['param_name']}")
        print(f"  比特位置: {cb['bit_position']}")
        print(f"  准确率下降: {cb['accuracy_drop']:.2f}%")
        print(f"  攻击后准确率: {cb['accuracy_after_attack']:.2f}%")
    
    # 执行攻击
    print("\n执行比特翻转攻击...")
    attacked_model, final_accuracy, flip_records = attacker.perform_attack(test_loader, critical_bits)
    
    print(f"\n攻击完成! 最终准确率: {final_accuracy:.2f}%")
    print(f"准确率下降: {original_accuracy - final_accuracy:.2f}%")
    
    # 保存攻击后的模型
    save_model(attacked_model, ATTACKED_MODEL_PATH)
    
    # 提取攻击后模型的参数
    attacked_params = extract_model_parameters(attacked_model)
    
    # 对比所有参数
    param_comparison = compare_parameters(original_params, attacked_params)
    
    # 打印对比结果
    print_parameter_comparison(param_comparison)
    
    # 打印翻转记录
    print("\n===== 比特翻转操作记录 =====")
    for i, record in enumerate(flip_records):
        print(f"翻转操作 #{i+1}:")
        print(f"  参数名称: {record['param_name']}")
        print(f"  元素索引(展平后): {record['element_index']}")
        print(f"  原始形状: {record['original_shape']}")
        print(f"  翻转的比特位置: {record['bit_position']}")
        print(f"  原始值: {record['original_value']:.6f}")

if __name__ == "__main__":
    main()
    