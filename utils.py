import cv2
import numpy as np
from matplotlib import pyplot as plt

def load_image(path):
    """加载图像"""
    return cv2.imread(path)

def save_image(image, path):
    """保存图像"""
    cv2.imwrite(path, image)

def show_image(image, title='Image', cmap=None):
    """显示图像（适用于Jupyter环境）"""
    if len(image.shape) == 2:  # 灰度图
        plt.imshow(image, cmap='gray')
    else:  # 彩色图
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

# ================= 图像增强 =================
def adjust_brightness(image, factor):
    """调整亮度"""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv = hsv.astype("float32")
    hsv[:, :, 2] = hsv[:, :, 2] * factor
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
    hsv = hsv.astype("uint8")
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def adjust_contrast(image, alpha):
    """调整对比度"""
    return cv2.convertScaleAbs(image, alpha=alpha)

def hist_equalization(image):
    """直方图均衡化"""
    if len(image.shape) == 2:  # 灰度图
        return cv2.equalizeHist(image)
    else:  # 彩色图
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        y_eq = cv2.equalizeHist(y)
        return cv2.cvtColor(cv2.merge((y_eq, cr, cb)), cv2.COLOR_YCrCb2BGR)

# ================= 图像分割 =================
def threshold_segmentation(image, threshold=127):
    """阈值分割"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return binary

def edge_detection(image, low=50, high=150):
    """Canny边缘检测"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    return cv2.Canny(gray, low, high)

# ================= 图像平滑 =================
def mean_blur(image, kernel_size=5):
    """均值滤波"""
    return cv2.blur(image, (kernel_size, kernel_size))

def gaussian_blur(image, kernel_size=5, sigma=0):
    """高斯滤波"""
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

def median_blur(image, kernel_size=5):
    """中值滤波"""
    return cv2.medianBlur(image, kernel_size)

# ================= 图像锐化 =================
def laplacian_sharpen(image):
    """拉普拉斯锐化"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    sharpened = gray - 0.5*laplacian
    return np.uint8(np.clip(sharpened, 0, 255))

    # ================= 图像恢复 =================

def denoising(image, method='fast_nlmeans', h=10, template_size=7, search_size=21):
    """
    图像去噪恢复
    参数:
        image: 输入图像
        method: 去噪方法 ('fast_nlmeans'或'gaussian')
        h: 滤波强度(仅对nlmeans有效)
        template_size: 模板窗口大小(仅对nlmeans有效)
        search_size: 搜索窗口大小(仅对nlmeans有效)
    返回:
        去噪后的图像
    """
    if len(image.shape) == 3:  # 彩色图像
        if method == 'fast_nlmeans':
            return cv2.fastNlMeansDenoisingColored(image, None, h, h, template_size, search_size)
        else:
            # 对各通道分别进行高斯去噪
            b, g, r = cv2.split(image)
            b = cv2.GaussianBlur(b, (5, 5), 0)
            g = cv2.GaussianBlur(g, (5, 5), 0)
            r = cv2.GaussianBlur(r, (5, 5), 0)
            return cv2.merge([b, g, r])
    else:  # 灰度图像
        if method == 'fast_nlmeans':
            return cv2.fastNlMeansDenoising(image, None, h, template_size, search_size)
        else:
            return cv2.GaussianBlur(image, (5, 5), 0)


def deblurring(image, kernel_size=(5, 5), method='wiener'):
    """
    图像去模糊恢复
    参数:
        image: 输入图像
        kernel_size: 模糊核大小
        method: 去模糊方法 ('wiener'或'inverse')
    返回:
        去模糊后的图像
    """
    # 模拟一个模糊核(实际应用中应该估计真实的模糊核)
    kernel = np.ones(kernel_size, np.float32) / (kernel_size[0] * kernel_size[1])

    if method == 'wiener':
        # Wiener滤波去模糊
        blurred = cv2.filter2D(image, -1, kernel)
        psf = kernel / np.sum(kernel)
        return cv2.filter2D(blurred, -1, np.rot90(psf, 2))
    else:
        # 逆滤波去模糊(简单实现)
        blurred = cv2.filter2D(image, -1, kernel)
        fft_blurred = np.fft.fft2(blurred)
        fft_kernel = np.fft.fft2(kernel, s=image.shape)
        epsilon = 1e-10  # 防止除以0
        fft_restored = fft_blurred / (fft_kernel + epsilon)
        restored = np.fft.ifft2(fft_restored)
        return np.abs(restored).astype(np.uint8)