import os
from PIL import Image
from torch.utils.data import Dataset
from configs import Config


class PrintToHandwritingDataset(Dataset):
    # 初始化训练集
    def __init__(self, print_dir, handwriting_dir, transform=None):
        self.print_dir = print_dir
        self.handwriting_dir = handwriting_dir
        self.transform = Config.transform

        # 确保文件顺序一致
        self.print_images = sorted([f for f in os.listdir(print_dir) if f.endswith(('.png', '.jpg'))])
        self.handwriting_images = sorted([f for f in os.listdir(handwriting_dir) if f.endswith(('.png', '.jpg'))])

        assert len(self.print_images) == len(self.handwriting_images), \
            f"数据集不匹配: 印刷体{len(self.print_images)}张, 手写体{len(self.handwriting_images)}张"

    def __len__(self):
        return len(self.print_images)

    # 数据获取方法(一对一）
    def __getitem__(self, idx):
        print_img = self._load_image(os.path.join(self.print_dir, self.print_images[idx]))
        hw_img = self._load_image(os.path.join(self.handwriting_dir, self.handwriting_images[idx]))
        return print_img, hw_img

    # 图像预处理
    def _load_image(self, path):
        img = Image.open(path).convert('L')  # 强制转为灰度
        if self.transform:
            img = self.transform(img)
        return img