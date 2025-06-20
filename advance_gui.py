import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import os
from handle_process import generate_character_images
from handle_img import create_pdf
from test import main as generate_handwriting
import tempfile


class HandwritingConverterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("印刷体转手写体工具")
        self.root.geometry("800x600")

        # 初始化变量
        self.output_dir = os.path.join(tempfile.gettempdir(), "HandwritingOutput")
        self.pdf_filename = "handwriting_output.pdf"

        # 创建输出目录
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        # 创建GUI界面
        self.create_widgets()

    def create_widgets(self):
        # 主框架
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 左侧面板 - 输入区域
        self.left_frame = tk.Frame(self.main_frame, width=500, bg='white')
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # 右侧面板 - 控制区域
        self.right_frame = tk.Frame(self.main_frame, width=200)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=5)

        # 输入框标签
        self.input_label = tk.Label(self.left_frame, text="输入文本:", font=('Arial', 12), bg='white')
        self.input_label.pack(anchor='w', pady=(0, 5))

        # 文本输入框
        self.text_input = tk.Text(self.left_frame, height=20, wrap=tk.WORD, font=('Arial', 11))
        self.text_input.pack(fill=tk.BOTH, expand=True)

        # 注意事项框架
        self.notes_frame = tk.Frame(self.left_frame, bd=1, relief=tk.SUNKEN, bg='#FFFFCC')
        self.notes_frame.pack(fill=tk.X, pady=(10, 0))

        # 注意事项标签
        self.notes_label = tk.Label(
            self.notes_frame,
            text="注意事项",
            font=('Arial', 10, 'bold'),
            bg='#FFFFCC'
        )
        self.notes_label.pack(anchor='w')

        # 注意事项内容
        notes_text = """1. 输入文本可包含中文、英文和标点符号
2. 生成的PDF将保留原始文本的排版
3. 标点符号会使用标准印刷体显示
4. 长文本可能需要较长时间处理"""
        self.notes_content = tk.Label(
            self.notes_frame,
            text=notes_text,
            justify=tk.LEFT,
            bg='#FFFFCC',
            font=('Arial', 9)
        )
        self.notes_content.pack(anchor='w', padx=5, pady=5)

        # 右侧控制面板
        self.control_frame = tk.LabelFrame(self.right_frame, text="控制面板", padx=5, pady=5)
        self.control_frame.pack(fill=tk.X, pady=5)

        # PDF文件名输入
        self.filename_frame = tk.Frame(self.control_frame)
        self.filename_frame.pack(fill=tk.X, pady=5)

        self.filename_label = tk.Label(self.filename_frame, text="PDF文件名:")
        self.filename_label.pack(anchor='w')

        self.filename_var = tk.StringVar(value=self.pdf_filename)
        self.filename_entry = tk.Entry(self.filename_frame, textvariable=self.filename_var)
        self.filename_entry.pack(fill=tk.X)

        # 保存路径设置
        self.path_frame = tk.Frame(self.control_frame)
        self.path_frame.pack(fill=tk.X, pady=5)

        self.path_label = tk.Label(self.path_frame, text="保存路径:")
        self.path_label.pack(anchor='w')

        self.path_var = tk.StringVar()
        self.path_entry = tk.Entry(self.path_frame, textvariable=self.path_var, state='readonly')
        self.path_entry.pack(fill=tk.X)

        self.btn_set_path = tk.Button(
            self.path_frame,
            text="浏览...",
            command=self.set_output_dir,
            width=8
        )
        self.btn_set_path.pack(pady=5)

        # 生成按钮
        self.btn_generate = tk.Button(
            self.control_frame,
            text="生成手写体PDF",
            command=self.generate_pdf,
            bg='#4CAF50',
            fg='white',
            font=('Arial', 10, 'bold')
        )
        self.btn_generate.pack(fill=tk.X, pady=10)

        # 状态栏
        self.status_var = tk.StringVar()
        self.status_var.set("就绪")
        self.status_bar = tk.Label(
            self.root,
            textvariable=self.status_var,
            bd=1,
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        # 初始化路径显示
        self.path_var.set(self.output_dir)

    def set_output_dir(self):
        """设置输出目录"""
        dir_path = filedialog.askdirectory(title="选择保存目录")
        if dir_path:
            self.output_dir = dir_path
            self.path_var.set(dir_path)
            self.status_var.set(f"保存路径设置为: {dir_path}")

    def generate_pdf(self):
        """生成手写体PDF"""
        text_content = self.text_input.get("1.0", tk.END).strip()
        if not text_content:
            messagebox.showerror("错误", "请输入要转换的文本!")
            return

        pdf_name = self.filename_var.get().strip() or "handwriting_output.pdf"
        if not pdf_name.endswith('.pdf'):
            pdf_name += '.pdf'

        output_path = os.path.join(self.output_dir, pdf_name)

        try:
            self.btn_generate.config(state=tk.DISABLED)
            self.status_var.set("正在生成手写体...")
            self.root.update()  # 强制更新界面

            # 1. 生成印刷体字符图片
            imgs, punc_idx = generate_character_images(text_content)

            # 2. 转换为手写体
            handle_imgs = generate_handwriting(imgs, punc_idx)

            # 3. 创建PDF
            create_pdf(handle_imgs, output_path)

            self.status_var.set(f"转换完成! PDF已保存到: {output_path}")
            messagebox.showinfo("完成", f"手写体PDF已生成:\n{output_path}")

        except Exception as e:
            messagebox.showerror("错误", f"生成过程中出错:\n{str(e)}")
            self.status_var.set("生成出错")
        finally:
            self.btn_generate.config(state=tk.NORMAL)


# 运行应用程序
if __name__ == "__main__":
    root = tk.Tk()
    app = HandwritingConverterApp(root)
    root.mainloop()