# 数字图像处理 - 基于GAN的印刷体转手写体处理系统

## 项目介绍

**基础部分**
  
主要展示各类基础图像处理算法的实际应用效果。该模块不直接参与印刷体转手写体的生成流程，而是为项目提供基础的图像处理能力验证。包含功能有：
  - 图像增强：亮度调整、对比度调整、直方图均衡化
  
  - 图像分割：阈值分割、边缘检测
  
  - 图像平滑：均值滤波、高斯滤波、中值滤波
  
  - 图像锐化：拉普拉斯锐化
  
  - 图像恢复：去噪、去模糊

**高级部分**
  
一个将印刷体文字转换为手写体风格的工具，基于深度学习技术实现。主要功能包括：
  - 由文本生成单字印刷体图片
    
  - 使用生成对抗网络(GAN)将印刷体转换为手写体风格
    
  - 保留原始文本的排版和标点符号
    
  - 通过图像处理将手写体缝合

## 环境依赖
  
### 系统要求
- Python 3.8+
- CUDA 11.3+ 
- PyTorch 1.10+

### Python库依赖
- torch>=1.10.0
- torchvision>=0.11.0 
- numpy>=1.21.0
- opencv-python>=4.5.0
- Pillow>=8.4.0
- fpdf2>=1.7.0
- tkinter

## 目录结构
    ├── advance_gui.py         # 高级功能的GUI界面
    
    ├── configs.py             # 训练模型配置参数
    
    ├── datasets.py            # 数据加载和处理
    
    ├── gui.py                 # 基础功能的GUI界面
    
    ├── handle_img.py          # 图像处理和PDF生成
    
    ├── handle_process.py      # 印刷体图像生成
    
    ├── main.py                # 基础图像处理示例
    
    ├── models.py              # 神经网络模型定义
    
    ├── README.md              # 项目说明
    
    ├──save_models             # 训练好的模型
    
    ├── test.py                # 印刷体转手写体功能
    
    └── utils.py               # 图像处理工具函数


## 使用指南
1.安装依赖
  
    pip install -r requirements.txt

2.运行基础功能

    python gui.py

3.运行高级功能

    python advance_gui.py

