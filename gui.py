import tempfile
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
from utils import *  # 导入之前写的图像处理函数


class ImageProcessorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("数字图像处理工具")
        self.root.geometry("800x600")

        # 初始化变量
        self.image_path = ""
        self.original_image = None
        self.processed_image = None
        self.output_dir = os.path.join(tempfile.gettempdir(), "ImageProcessorOutput")
        self.custom_filename = ""  # 自定义文件名

        # 创建输出目录
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        # 创建GUI界面
        self.create_widgets()

    def create_widgets(self):
        # 左侧面板 - 图像显示
        self.left_frame = tk.Frame(self.root, width=400, height=500, bg='white')
        self.left_frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)

        # 原始图像显示
        self.original_label = tk.Label(self.left_frame, text="原始图像", bg='white')
        self.original_label.pack()
        self.original_panel = tk.Label(self.left_frame, bg='white')
        self.original_panel.pack(fill=tk.BOTH, expand=True)

        # 处理后图像显示
        self.processed_label = tk.Label(self.left_frame, text="处理后图像", bg='white')
        self.processed_label.pack()
        self.processed_panel = tk.Label(self.left_frame, bg='white')
        self.processed_panel.pack(fill=tk.BOTH, expand=True)

        # 右侧面板 - 控制面板
        self.right_frame = tk.Frame(self.root, width=200, height=500)
        self.right_frame.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.Y)

        # 选择图像按钮
        self.btn_open = tk.Button(self.right_frame, text="选择图像", command=self.open_image)
        self.btn_open.pack(fill=tk.X, pady=5)

        # 处理方式选择
        self.process_label = tk.Label(self.right_frame, text="选择处理方式:")
        self.process_label.pack(fill=tk.X, pady=(10, 0))

        self.process_var = tk.StringVar()
        self.process_combobox = ttk.Combobox(self.right_frame, textvariable=self.process_var, state="readonly")
        self.process_combobox['values'] = (
            "图像增强 - 亮度调整",
            "图像增强 - 对比度调整",
            "图像增强 - 直方图均衡化",
            "图像分割 - 阈值分割",
            "图像分割 - 边缘检测",
            "图像平滑 - 均值滤波",
            "图像平滑 - 高斯滤波",
            "图像锐化 - 拉普拉斯",
            "图像恢复 - 去噪",
            "图像恢复 - 去模糊"
        )
        self.process_combobox.pack(fill=tk.X, pady=5)
        self.process_combobox.current(0)

        # 参数调整滑块
        self.param_frame = tk.Frame(self.right_frame)
        self.param_frame.pack(fill=tk.X, pady=5)

        self.param_label = tk.Label(self.param_frame, text="参数值:")
        self.param_label.pack(side=tk.LEFT)

        self.param_var = tk.DoubleVar(value=1.0)
        self.param_scale = tk.Scale(self.param_frame, from_=0.1, to=3.0, resolution=0.1,
                                    orient=tk.HORIZONTAL, variable=self.param_var)
        self.param_scale.pack(fill=tk.X, expand=True)

        # 处理按钮
        self.btn_process = tk.Button(self.right_frame, text="处理图像", command=self.process_image)
        self.btn_process.pack(fill=tk.X, pady=5)

        # 设置保存路径按钮
        self.btn_set_path = tk.Button(self.right_frame, text="设置保存路径", command=self.set_output_dir)
        self.btn_set_path.pack(fill=tk.X, pady=5)

        # 自定义文件名输入框
        self.filename_frame = tk.Frame(self.right_frame)
        self.filename_frame.pack(fill=tk.X, pady=5)

        self.filename_label = tk.Label(self.filename_frame, text="文件名:")
        self.filename_label.pack(side=tk.LEFT)

        self.filename_var = tk.StringVar()
        self.filename_entry = tk.Entry(self.filename_frame, textvariable=self.filename_var)
        self.filename_entry.pack(fill=tk.X, expand=True)

        # 保存按钮
        self.btn_save = tk.Button(self.right_frame, text="保存结果", command=self.save_image)
        self.btn_save.pack(fill=tk.X, pady=5)

        # 状态栏
        self.status_var = tk.StringVar()
        self.status_var.set("就绪")
        self.status_bar = tk.Label(self.root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def open_image(self):
        """打开图像文件"""
        file_path = filedialog.askopenfilename(
            title="选择图像文件",
            filetypes=[("图像文件", "*.jpg;*.jpeg;*.png;*.bmp"), ("所有文件", "*.*")]
        )

        if file_path:
            self.image_path = file_path
            self.original_image = cv2.imread(file_path)

            # 显示原始图像
            self.show_image(self.original_image, self.original_panel)
            self.status_var.set(f"已加载: {os.path.basename(file_path)}")

    def set_output_dir(self):
        """设置输出目录"""
        dir_path = filedialog.askdirectory(title="选择保存目录")
        if dir_path:
            self.output_dir = dir_path
            self.status_var.set(f"保存路径设置为: {dir_path}")

    def process_image(self):
        """处理图像"""
        if self.original_image is None:
            messagebox.showerror("错误", "请先选择图像!")
            return

        process_type = self.process_var.get()
        param = self.param_var.get()

        try:
            if "亮度调整" in process_type:
                self.processed_image = adjust_brightness(self.original_image, param)
            elif "对比度调整" in process_type:
                self.processed_image = adjust_contrast(self.original_image, param)
            elif "直方图均衡化" in process_type:
                self.processed_image = hist_equalization(self.original_image)
            elif "阈值分割" in process_type:
                self.processed_image = threshold_segmentation(self.original_image, int(param * 127))
            elif "边缘检测" in process_type:
                self.processed_image = edge_detection(self.original_image, int(param * 50), int(param * 150))
            elif "均值滤波" in process_type:
                kernel_size = int(param * 10)
                kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size  # 确保奇数
                self.processed_image = mean_blur(self.original_image, kernel_size)
            elif "高斯滤波" in process_type:
                kernel_size = int(param * 10)
                kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size  # 确保奇数
                self.processed_image = gaussian_blur(self.original_image, kernel_size)
            elif "拉普拉斯" in process_type:
                self.processed_image = laplacian_sharpen(self.original_image)
            elif "去噪" in process_type:
                self.processed_image = denoising(self.original_image, h=int(param * 20))
            elif "去模糊" in process_type:
                self.processed_image = deblurring(self.original_image, (int(param * 5), int(param * 5)))

            # 显示处理后的图像
            self.show_image(self.processed_image, self.processed_panel)
            self.status_var.set(f"已完成: {process_type}")

        except Exception as e:
            messagebox.showerror("处理错误", f"图像处理时出错:\n{str(e)}")
            self.status_var.set("处理出错")

    def save_image(self):
        """保存处理后的图像"""
        if self.processed_image is None:
            messagebox.showerror("错误", "没有可保存的处理结果!")
            return

        # 获取自定义文件名
        custom_name = self.filename_var.get().strip()

        # 生成保存路径
        original_name = os.path.splitext(os.path.basename(self.image_path))[0]
        process_type = self.process_var.get().split("-")[-1].strip()

        if custom_name:
            # 使用自定义文件名
            save_name = f"{custom_name}.jpg"
        else:
            # 使用默认文件名
            save_name = f"{original_name}_{process_type}.jpg"

        save_path = os.path.join(self.output_dir, save_name)

        # 确保目录存在
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # 保存图像
        try:
            cv2.imwrite(save_path, self.processed_image)
            self.status_var.set(f"已保存: {save_path}")
            messagebox.showinfo("保存成功", f"图像已保存到:\n{save_path}")
        except Exception as e:
            messagebox.showerror("保存错误", f"保存图像时出错:\n{str(e)}")
            self.status_var.set("保存出错")

    def show_image(self, image, panel):
        """在指定面板显示图像"""

        # 调整图像大小以适应面板
        max_width = self.left_frame.winfo_width() - 20
        max_height = self.left_frame.winfo_height() // 2 - 20

        # 转换颜色空间(BGR to RGB)
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # 计算缩放比例
        h, w = image.shape[:2]
        ratio = min(max_width / w, max_height / h)
        new_w, new_h = int(w * ratio), int(h * ratio)

        # 调整大小
        image = cv2.resize(image, (new_w, new_h))

        # 转换为PhotoImage
        from PIL import Image
        img_pil = Image.fromarray(image)
        img_tk = ImageTk.PhotoImage(img_pil)

        # 更新面板
        panel.configure(image=img_tk)
        panel.image = img_tk  # 保持引用
        setattr(self, f"{panel.winfo_name()}_photo", img_tk)


# 运行应用程序
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessorApp(root)
    root.mainloop()