import onnxruntime as ort
import numpy as np

try:
    # 強制指定使用 CUDA
    session = ort.InferenceSession("test.onnx", providers=['CUDAExecutionProvider'])
    # (如果沒有 test.onnx，這行會噴找不到檔案，但如果是噴 LoadLibrary Error 126，那就是環境變數問題)
    print("🚀 恭喜！CUDA 引擎已成功載入並運作。")
except Exception as e:
    print(f"❌ 載入失敗。錯誤訊息：{e}")

import os
import subprocess
import glob
import cv2
import numpy as np
import json

# --- 配置區 ---
INPUT_FRAMES = "D:/NCKU/CSIE-Project/Colla/input_video_frames/" # 放原始剪影序列的資料夾
OUTPUT_BASE = "D:/NCKU/CSIE-Project/Colla/batch_results/"
CONTROL_MAPS_DIR = os.path.join(OUTPUT_BASE, "control_maps")
os.makedirs(CONTROL_MAPS_DIR, exist_ok=True)

def run_icas_pipeline():
    # 1. 取得所有輸入影格 (例如 frame_001.png, frame_002.png...)
    frames = sorted(glob.glob(os.path.join(INPUT_FRAMES, "*.png")))
    print(f"找到 {len(frames)} 個影格，開始執行 ICAS 幾何最佳化...")

    for i, frame_path in enumerate(frames):
        frame_name = os.path.basename(frame_path).replace(".png", "")
        output_dir = os.path.join(OUTPUT_BASE, frame_name)
        os.makedirs(output_dir, exist_ok=True)

        print(f"正在處理影格 [{i+1}/{len(frames)}]: {frame_name}")

        # 步驟 1: 執行形狀分解 (MAD) [cite: 1091, 1229]
        # 產出 final_cut.json
        subprocess.run(["python", "shape_decomposition.py", frame_path, output_dir], check=True)

        # 步驟 2: 執行形狀感知切片 (SAS) 與最佳化 [cite: 1097, 1205]
        # 產出 slicing_result.json
        # 注意：這裡需要指向你的圖片集遮罩路徑 (mask)
        subprocess.run(["python", "sas_optimization.py", frame_path, "./resources/mask", output_dir], check=True)

        # 步驟 3: 渲染 ControlNet 用的線稿 (Edge Map)
        json_path = os.path.join(output_dir, "slicing_result.json")
        render_edge_map(json_path, CONTROL_MAPS_DIR, frame_name)

def render_edge_map(json_path, output_dir, name):
    with open(json_path, 'r') as f:
        data = json.load(f)
    canvas = np.zeros((data['height'], data['width'], 3), dtype=np.uint8)
    for part in data.get('parts', []):
        pts = np.array(part['polygon'], dtype=np.int32)
        cv2.polylines(canvas, [pts], True, (255, 255, 255), 2)
    cv2.imwrite(os.path.join(output_dir, f"{name}_edge.png"), canvas)

if __name__ == "__main__":
    run_icas_pipeline()