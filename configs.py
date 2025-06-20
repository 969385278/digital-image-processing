import torch
import torchvision


class Config:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 训练参数
    image_size = 100
    batch_size = 32
    lr = 0.0002
    num_epochs = 200
    lambda_content = 100
    print_every = 100
    save_every = 10
    num_workers = 4
    channels = 1  # 灰度图像
    style_dim = 5  # 可选风格维度


    # 数据预处理配置
    transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((image_size, image_size)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.5], std=[0.5])
    ])

# 路径配置
paths = {
    'print_train': 'data/train/print',
    'handwriting_train': 'data/train/handwriting',
    'print_test': 'data/test/print',
    'output': 'results',
    'saved_models': 'saved_models',
    'samples': 'images'
}