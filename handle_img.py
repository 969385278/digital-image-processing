import cv2
import numpy as np
from PIL import Image
from fpdf import FPDF
import os

from handle_process import generate_character_images
from test import main, get_multiline_input

# 重新定义工具类里面的高斯滤波
def gaussian_blur_binary(image, kernel_size=5, sigma=0):
    # 转换为 NumPy 数组
    img_np = np.array(image)

    # 高斯模糊（输入必须是 8-bit 或 float）
    blurred = cv2.GaussianBlur(img_np, (kernel_size, kernel_size), sigma)

    # 转换回 PIL（模式 'L'）
    return Image.fromarray(blurred)

def create_pdf(images, output_path):
    pdf = FPDF()
    pdf.add_page()
    # 参数调整
    img_size_mm = 8
    margin = 15
    h_spacing = 0
    v_spacing = 0
    page_width = 210

    x, y = margin, margin


    for i, img in enumerate(images):
        # 转换为PIL Image
        if isinstance(img, np.ndarray):
            img_pil = Image.fromarray(img.astype('uint8'), 'L')
        else:
            img_pil = img

        # 临时文件处理（使用绝对路径）
        temp_dir = os.path.abspath("temp")
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, f"temp_{i}.jpg")

        try:
            img_pil.save(temp_path)

            # 坐标计算（简化逻辑）
            if x + img_size_mm > page_width - margin:
                x = margin
                y += img_size_mm + v_spacing

                if y > 270:  # 留出底部边距
                    pdf.add_page()
                    y = margin

            # 添加图像
            pdf.image(temp_path, x, y, img_size_mm)
            x += img_size_mm + h_spacing

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    pdf.output(output_path)

if __name__ == '__main__':

    content = get_multiline_input()
    imgs, punc_idx = generate_character_images(content)
    handle_imgs = main(imgs, punc_idx)
    handled_imgs = [gaussian_blur_binary(img) for img in handle_imgs]
    # 生成PDF
    output_pdf = "output.pdf"
    create_pdf(handled_imgs, output_pdf)

    print(f"PDF已生成: {output_pdf}")