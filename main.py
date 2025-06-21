import cv2
import numpy as np
from utils import *


def main():
    # 1. 加载图像
    image_path = "C:/Users/lenovo/Desktop/project test/123.jpg"
    image = load_image(image_path)

    # 2. 图像增强示例
    bright_image = adjust_brightness(image, 1.5)  # 增加亮度50%
    contrast_image = adjust_contrast(image, 1.5)  # 增加对比度50%
    eq_image = hist_equalization(image)  # 直方图均衡化

    # 3. 图像分割示例
    binary_image = threshold_segmentation(image, 120)  # 阈值分割
    edges = edge_detection(image)  # 边缘检测

    # 4. 图像平滑示例
    mean_blurred = mean_blur(image, 7)  # 均值滤波
    gaussian_blurred = gaussian_blur(image, 7)  # 高斯滤波

    # 5. 图像锐化示例
    sharpened_laplace = laplacian_sharpen(image)  # 拉普拉斯锐化

    # 6. 图像恢复示例
    # 6.1 添加噪声(为演示去噪效果)
    noisy_image = image.copy()
    if len(noisy_image.shape) == 2:  # 灰度图
        noisy_image = cv2.add(noisy_image, np.random.normal(0, 25, noisy_image.shape).astype(np.uint8))
    else:  # 彩色图
        noise = np.random.normal(0, 25, noisy_image.shape).astype(np.uint8)
        noisy_image = cv2.add(noisy_image, noise)

    # 6.2 添加模糊(为演示去模糊效果)
    kernel = np.ones((7, 7), np.float32) / 49
    blurred_image = cv2.filter2D(image, -1, kernel)

    # 6.3 图像恢复处理
    denoised_image = denoising(noisy_image, method='fast_nlmeans')
    deblurred_image = deblurring(blurred_image, method='wiener')

    # 7. 保存结果
    save_image(bright_image, "C:/Users/lenovo/Desktop/project test/bright.jpg")
    save_image(contrast_image, "C:/Users/lenovo/Desktop/project test/contrast.jpg")
    save_image(eq_image, "C:/Users/lenovo/Desktop/project test/equalized.jpg")
    save_image(binary_image, "C:/Users/lenovo/Desktop/project test/binary.jpg")
    save_image(edges, "C:/Users/lenovo/Desktop/project test/edges.jpg")
    save_image(mean_blurred, "C:/Users/lenovo/Desktop/project test/mean_blur.jpg")
    save_image(gaussian_blurred, "C:/Users/lenovo/Desktop/project test/gaussian_blur.jpg")
    save_image(sharpened_laplace, "C:/Users/lenovo/Desktop/project test/sharpened_laplace.jpg")
    save_image(noisy_image, "C:/Users/lenovo/Desktop/project test/noisy.jpg")
    save_image(blurred_image, "C:/Users/lenovo/Desktop/project test/blurred.jpg")
    save_image(denoised_image, "C:/Users/lenovo/Desktop/project test/denoised.jpg")
    save_image(deblurred_image, "C:/Users/lenovo/Desktop/project test/deblurred.jpg")
    print("所有处理完成！结果保存在C:/Users/lenovo/Desktop/project test目录")


if __name__ == "__main__":
    main()
