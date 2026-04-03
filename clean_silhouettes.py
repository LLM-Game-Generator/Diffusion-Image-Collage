import cv2
import os
import glob
import numpy as np

input_dir = "./input_silhouette_frames"
output_dir = "./input_silhouette_frames"
os.makedirs(output_dir, exist_ok=True)

# 抓取所有 png
frame_files = glob.glob(os.path.join(input_dir, "*.png"))
print(f"準備清洗 {len(frame_files)} 張剪影...")

for filepath in frame_files:
    filename = os.path.basename(filepath)
    # 讀取為灰階
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

    # 1. 確保影像是二值化 (黑底白字或白底黑字)
    # 這裡我們將黑色的剪影反轉成白色，方便找輪廓 (OpenCV 預設找白色物體)
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    # 2. 尋找所有輪廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 建立一張全白的新畫布 (這就是 ICAS 喜歡的白底)
    clean_img = np.ones_like(img) * 255

    if contours:
        # 3. 找出面積最大的輪廓 (也就是人物主體，丟棄所有懸浮雜訊)
        main_contour = max(contours, key=cv2.contourArea)

        # 4. 在白底上畫出最大輪廓，並用黑色填滿
        cv2.drawContours(clean_img, [main_contour], -1, 0, thickness=cv2.FILLED)

    # 存檔
    out_path = os.path.join(output_dir, filename)
    cv2.imwrite(out_path, clean_img)

print("✨ 清洗完成！雜訊與破洞已全數殲滅。")