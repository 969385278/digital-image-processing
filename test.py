import os
import sys

import cv2
import numpy as np
import torch

from handle_process import generate_character_images
from models import Generator
from configs import Config, paths

def get_multiline_input():
    print("请粘贴文本（含标点和换行符，Ctrl+Z+回车结束）：")
    return sys.stdin.read()  # 保留所有原始符号
def main(imgs, punc_idx):
    # 初始化模型
    generator = Generator(Config.channels, Config.channels).to(Config.device)
    generator.load_state_dict(torch.load(os.path.join(paths['saved_models'], 'generator_110.pth'),weights_only=True ))
    generator.eval()

    # 创建输出目录
    #os.makedirs(paths['output'], exist_ok=True)
    i = 0
    #print(punc_idx)
    # 处理测试图像
    handle_imgs = []
    for img in imgs:
        i+=1
        if i in punc_idx:
            """output_path = os.path.join(paths['output'], f'{i}.jpg')
            img.save(output_path)"""
            handle_imgs.append(img)
            #print(f"[标点保存] {output_path}")
        else:
            # 加载图像
            img = img.convert('L')
            img_tensor = Config.transform(img).unsqueeze(0).to(Config.device)

            # 生成结果
            with torch.no_grad():
                generated = generator(img_tensor)

            # 保存结果
            output = (generated.squeeze().cpu().numpy() + 1) * 127.5  # [-1,1] -> [0,255]
            """output_path = os.path.join(paths['output'], f'{i}.jpg')
            cv2.imwrite(output_path, output.astype(np.uint8))"""
            handle_imgs.append(output.astype(np.uint8))
            #print(f"Generated: {output_path}")
    #print("已完成转换")
    return handle_imgs


if __name__ == '__main__':

    content = get_multiline_input()

    imgs, punc_idx =  generate_character_images(content)
    handle_imgs = main(imgs,punc_idx)

    output_dir = r"C:\Users\86182\Desktop\hp"
    os.makedirs(output_dir, exist_ok=True)

    for idx, img in enumerate(handle_imgs,start=1):
        output_path = os.path.join(output_dir, f"{idx}.jpg")
        if isinstance(img, np.ndarray):  # 手写体(numpy数组)
            cv2.imwrite(output_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        else:  # 标点(PIL Image)
            img.save(output_path)
        print(f"已保存: {output_path}")

    print(f"所有图片已保存到: {output_dir}")