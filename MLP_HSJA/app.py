import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageTk
import tkinter as tk
from tkinter import messagebox
from main import MLP  # 从训练文件中导入模型定义

class DigitRecognizer:
    """实时手写数字识别GUI应用"""
    def __init__(self, model_path):
        # 设置设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # 加载模型
        self.model = MLP().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print(f"Model loaded from {model_path}")
        
        # 创建GUI窗口
        self.window = tk.Tk()
        self.window.title("手写数字识别")
        
        # 创建画布
        self.canvas = tk.Canvas(self.window, width=280, height=280, bg="black")
        self.canvas.pack(pady=20)
        
        # 创建按钮框架
        button_frame = tk.Frame(self.window)
        button_frame.pack(fill=tk.X, padx=50)
        
        # 创建识别按钮
        self.recognize_btn = tk.Button(
            button_frame, text="识别", command=self.recognize_digit, width=10)
        self.recognize_btn.pack(side=tk.LEFT, padx=10)
        
        # 创建清除按钮
        self.clear_btn = tk.Button(
            button_frame, text="清除", command=self.clear_canvas, width=10)
        self.clear_btn.pack(side=tk.LEFT, padx=10)
        
        # 创建保存按钮
        self.save_btn = tk.Button(
            button_frame, text="保存图像", command=self.save_image, width=10)
        self.save_btn.pack(side=tk.RIGHT, padx=10)
        
        # 结果显示标签
        self.result_label = tk.Label(
            self.window, text="请绘制数字", font=("Arial", 24))
        self.result_label.pack(pady=20)
        
        # 绑定鼠标事件
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset)
        self.image = Image.new("L", (280, 280), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.last_x, self.last_y = None, None
        
    def paint(self, event):
        """处理鼠标绘制事件"""
        x, y = event.x, event.y
        r = 15  # 画笔半径
        
        if self.last_x and self.last_y:
            # 绘制线条连接点
            self.canvas.create_line(
                self.last_x, self.last_y, x, y, 
                width=2*r, fill="white", capstyle=tk.ROUND, smooth=tk.TRUE
            )
            self.draw.line(
                [self.last_x, self.last_y, x, y], 
                fill=255, width=2*r
            )
        
        # 绘制当前点
        self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="white", outline="white")
        self.draw.ellipse([x-r, y-r, x+r, y+r], fill=255)
        
        self.last_x, self.last_y = x, y
    
    def reset(self, event):
        """重置最后绘制的点"""
        self.last_x, self.last_y = None, None
    
    def clear_canvas(self):
        """清除画布内容"""
        self.canvas.delete("all")
        self.image = Image.new("L", (280, 280), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="请绘制数字")
        self.last_x, self.last_y = None, None
    
    def save_image(self):
        """保存绘制的图像"""
        filename = "handwritten_digit.png"
        self.image.save(filename)
        messagebox.showinfo("保存成功", f"图像已保存为 {filename}")
    
    def recognize_digit(self):
        """识别绘制的手写数字"""
        # 将图像转换为MNIST格式
        img = self.image.resize((28, 28))  # 缩小到28x28
        img = np.array(img)
        img = img.astype(np.float32) / 255.0
        img = (img - 0.1307) / 0.3081  # MNIST相同的归一化
        
        # 转换为PyTorch张量
        img_tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(self.device)
        
        # 模型预测
        with torch.no_grad():
            output = self.model(img_tensor)
            probabilities = torch.softmax(output, dim=1)
            pred = output.argmax(dim=1, keepdim=True).item()
            confidence = probabilities[0, pred].item() * 100
        
        # 显示结果
        self.result_label.config(text=f"识别结果: {pred} (置信度: {confidence:.1f}%)")
    
    def run(self):
        """启动GUI应用"""
        self.window.mainloop()

if __name__ == '__main__':
    # 加载训练好的模型
    model_path = "mnist_mlp_best.pth"
    
    # 启动应用
    print("启动实时手写数字识别应用...")
    app = DigitRecognizer(model_path)
    app.run()