from PIL import Image, ImageDraw, ImageFont
import os


def generate_character_images(text):
    # 图片参数设置
    img_size = (100, 100)  # 图片大小
    bg_color = (255, 255, 255)  # 白色背景
    text_color = (0, 0, 0)  # 黑色文字

    font = ImageFont.truetype( "simsun.ttc", 80)

    punctuation_offset = {
        "，": -30, "。": -30, "、": -30,
        "！": -17, "？": -17, "；": -30,
        "：": -30, "“": -25, "”": -25,
        "（": -10, "）": -10, "《": -10,
        "》": -10, ".": -30, ",": -30,
        "!": -17, "?": -17, ";":-30,
        ":":-30,
    }
    punc_index = []
    imgs = []

    for idx, char in enumerate(text, start=1):
        if not char.strip():
            continue
        img = Image.new('RGB', img_size, bg_color)
        draw = ImageDraw.Draw(img)

        bbox = draw.textbbox((0, 0), char, font=font)
        text_width = bbox[2] - bbox[0]  # 右边界-左边界
        text_height = bbox[3] - bbox[1]  # 下边界-上边界

        # 计算居中位置
        x = (img_size[0] - text_width) / 2
        y = (img_size[1] - text_height) / 2

        if char in punctuation_offset:
            y += punctuation_offset[char]
            punc_index.append(idx)

        draw.text((x, y), char, fill=text_color, font=font)

        imgs.append(img)

        # 保存图片（按顺序命名）
        #img.save(os.path.join("E:\\pycharm_project\\handle_image", f"{idx}.jpg"), "JPEG", quality=95)
        #print(f"生成: {idx}.png ({char})")
    return imgs, punc_index


if __name__ == "__main__":
    # 使用原始字符串避免转义问题
    input_file = r"E:\pycharm_project\test1"
    output_dir = r"E:\pycharm_project\handle_image"
    # 可选：指定字体文件（如微软雅黑）
    # font_path = r"C:\Windows\Fonts\msyh.ttc"

    generate_character_images(input_file, output_dir)