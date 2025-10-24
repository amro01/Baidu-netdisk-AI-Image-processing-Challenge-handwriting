import cv2
import numpy as np
import os
import random
from PIL import Image

def overlay_transparent_png(background_img, overlay_png, position):
    """
    将一个带透明通道的PNG图像叠加到背景图像上。
    """
    if overlay_png.mode != 'RGBA':
        overlay_png = overlay_png.convert('RGBA')
    
    temp_layer = Image.new('RGBA', background_img.size, (0, 0, 0, 0))
    temp_layer.paste(overlay_png, position, mask=overlay_png)
    
    composite = Image.alpha_composite(background_img.convert('RGBA'), temp_layer)
    return composite.convert('RGB')


def main():
    """主函数，执行图像增强流程"""
    # --- 1. 设置路径 ---
    base_dir = 'tran_add_up'
    image_dir = os.path.join(base_dir, 'images')
    gts_dir = os.path.join(base_dir, 'gts')
    stroke_dir = os.path.join(base_dir, 'strokes')
    output_dir = os.path.join(base_dir, 'augmented_images')
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # --- 2. 准备文件列表 ---
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    
    # 分类大尺寸和小尺寸的涂抹素材
    large_strokes = [os.path.join(stroke_dir, f) for f in os.listdir(stroke_dir) if f.startswith('b') and f.endswith('.png')]
    small_strokes = [os.path.join(stroke_dir, f) for f in os.listdir(stroke_dir) if f.startswith('s') and f.endswith('.png')]

    if not image_files:
        print("错误：'images' 文件夹中没有找到 JPG 文件。")
        return
    if not large_strokes and not small_strokes:
        print("错误：'strokes' 文件夹中没有找到 PNG 文件。")
        return

    print(f"找到 {len(image_files)} 张图像进行处理...")
    print(f"加载了 {len(large_strokes)} 个大尺寸涂抹素材和 {len(small_strokes)} 个小尺寸涂抹素材。")

    for i, filename in enumerate(image_files):
        print(f"正在处理第 {i+1}/{len(image_files)} 张图像: {filename}")
        
        image_path = os.path.join(image_dir, filename)
        gts_path = os.path.join(gts_dir, filename.replace('.jpg', '.png'))

        if not os.path.exists(gts_path):
            print(f"警告：找不到对应的gts文件 {gts_path}，跳过此图像。")
            continue

        # --- 3. 定位手写痕迹 ---
        img = cv2.imread(image_path)
        gts_img = cv2.imread(gts_path)
        if img is None or gts_img is None:
            print(f"警告：读取图像 {filename} 或其gts时出错，跳过。")
            continue
        
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gts_gray = cv2.cvtColor(gts_img, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(img_gray, gts_gray)
        _, mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3,3),np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # --- 4. 识别笔迹轮廓 ---
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_contour_area = 5
        valid_contours = [c for c in contours if cv2.contourArea(c) > min_contour_area]
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # --- 5. 随机选择并应用涂抹 ---
        if valid_contours:
            num_contours_to_cover = int(len(valid_contours) * 0.8)
            if num_contours_to_cover == 0 and len(valid_contours) > 0:
                num_contours_to_cover = 1
            contours_to_cover = random.sample(valid_contours, num_contours_to_cover)
            
            for contour in contours_to_cover:
                x, y, w, h = cv2.boundingRect(contour)
                target_w, target_h = w, h
                
                # 优先使用大尺寸涂抹进行裁切
                use_large_stroke = random.choice([True, False]) if large_strokes and small_strokes else bool(large_strokes)
                stroke_to_process = None

                if use_large_stroke and large_strokes:
                    stroke_path = random.choice(large_strokes)
                    stroke_png = Image.open(stroke_path).convert('RGBA')
                    
                    # 随机裁切
                    if stroke_png.width > target_w and stroke_png.height > target_h:
                        crop_x = random.randint(0, stroke_png.width - target_w)
                        crop_y = random.randint(0, stroke_png.height - target_h)
                        stroke_to_process = stroke_png.crop((crop_x, crop_y, crop_x + target_w, crop_y + target_h))
                
                # 如果未使用大尺寸涂抹（或裁切失败），则使用小尺寸涂抹并缩放
                if stroke_to_process is None:
                    if not small_strokes: continue # 没有小图了，跳过
                    stroke_path = random.choice(small_strokes)
                    stroke_png = Image.open(stroke_path).convert('RGBA')
                    
                    scale = random.uniform(0.8, 1.5)
                    # 增加尺寸上限，例如250像素
                    max_size = 250
                    new_size = min(int(max(target_w, target_h) * scale), max_size)
                    if new_size > 0:
                        stroke_to_process = stroke_png.resize((new_size, new_size), Image.Resampling.LANCZOS)

                if stroke_to_process:
                    angle = random.uniform(0, 360)
                    stroke_rotated = stroke_to_process.rotate(angle, expand=True, resample=Image.Resampling.BICUBIC)
                    paste_x = x + (w - stroke_rotated.width) // 2
                    paste_y = y + (h - stroke_rotated.height) // 2
                    pil_img = overlay_transparent_png(pil_img, stroke_rotated, (paste_x, paste_y))
        else:
            print(f"警告：在 {filename} 中未找到有效的手写痕迹轮廓，将保存原图。")
            
        # --- 6. 保存结果 ---
        output_filename = filename.replace('.jpg', '_augmented.png')
        output_path = os.path.join(output_dir, output_filename)
        pil_img.save(output_path, 'PNG')

    print("处理完成！所有增强后的图像已保存到:", output_dir)


if __name__ == '__main__':
    main()