import cv2
import os
import numpy as np
from rembg import remove, new_session
from PIL import Image

# --- 強健 Session 初始化 ---
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
try:
    session = new_session("u2net", providers=providers)
    print("🚀 已啟動去背引擎 (優先使用 GPU)")
except Exception as e:
    print(f"⚠️ GPU 初始化失敗: {e}, 切換至純 CPU 模式")
    session = new_session("u2net", providers=['CPUExecutionProvider'])

# --- 設定路徑 ---
VIDEO_PATH = "./videos/82593-580137776_medium.mp4"
# 建議使用完整路徑避免權限問題
OUTPUT_DIR = "D:/NCKU/CSIE-Project/Colla/input_silhouette_frames/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

cap = cv2.VideoCapture(VIDEO_PATH)
frame_count = 0

print(f"🎬 開始處理影片: {VIDEO_PATH}")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 每 3 影格處理一次 (節省計算量)
    if frame_count % 3 == 0:
        # 1. 轉換色彩空間
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 2. 執行去背 (回傳帶有 Alpha 通道的 RGBA 陣列)
        # rembg 的 output 是一個 [H, W, 4] 的 numpy array
        output_rgba = remove(img_rgb, session=session)

        # 3. 提取 Alpha 通道 (第 4 層)
        alpha = output_rgba[:, :, 3]

        # 4. 二值化處理 (去除邊緣發光/半透明雜訊)
        # 只要透明度大於 100 就視為人體 (0)，否則視為背景 (255)
        # 這裡產出的是 ICAS 最愛的「白底黑人」格式
        silhouette = np.where(alpha > 100, 0, 255).astype(np.uint8)

        # 5. 轉回 Pillow 並儲存
        # 這裡我們存成 RGB 格式，確保沒有隱藏的透明層干擾 ICAS 讀取
        final_img = Image.fromarray(silhouette).convert("RGB")

        save_path = os.path.join(OUTPUT_DIR, f"frame_{frame_count:04d}.png")
        final_img.save(save_path)

        if frame_count % 30 == 0:
            print(f"⏳ 進度: 已處理第 {frame_count} 影格")

    frame_count += 1

cap.release()
print(f"✨ 去背與二值化完成！")
print(f"📂 檔案存放於: {OUTPUT_DIR}")
print(f"💡 提示：請確認生成的圖片為『純白背景、純黑剪影』，這將使 ICAS 運算速度提升 3 倍。")