import os
import glob
import time
import concurrent.futures
import src.icas.sas_optimization as so
import src.icas.shape_decomposition as sd
from generate_control_maps import generate_control_maps
import src.icas.collage_assembly as ca

# 設定路徑
INPUT_DIR = r"D:\NCKU\CSIE-Project\Colla\input_silhouette_frames\\"
JSON_OUT_DIR = r"D:\NCKU\CSIE-Project\Colla\batch_output\jsons\\"
IMAGE_COLLECTION_DIR = r"D:\NCKU\CSIE-Project\Colla\assets\images\\"
MASK_DIR = r"D:\NCKU\CSIE-Project\Colla\assets\mask\\"
CONTROL_MAPS_DIR = "./batch_output/control_maps"


def process_single_frame(frame_path):
    """
    單一影格的處理單元
    """
    try:
        frame_name = os.path.basename(frame_path).split(".")[0]
        frame_json_dir = os.path.join(JSON_OUT_DIR, frame_name) + os.sep

        if not os.path.exists(frame_json_dir):
            os.makedirs(frame_json_dir)

        sd.generate_cuts(frame_path, frame_json_dir)
        so.optimization(frame_path, MASK_DIR, frame_json_dir, True)
        # Render collage
        ca.render_collage(IMAGE_COLLECTION_DIR, frame_json_dir, 2)

        # Step 3: 產生控制圖
        # json_path = os.path.join(frame_json_dir, "slicing_result.json")
        # if os.path.exists(json_path):
        #     target_filename = f"control_{frame_name}.png"
        #     generate_control_maps(
        #         json_path,
        #         CONTROL_MAPS_DIR,
        #         scaling_factor=1.0,
        #         mode='edge',
        #         output_filename=target_filename
        #     )
        #     return f"✅ {frame_name} 完成"
        # else:
        #     return f"❌ {frame_name} 失敗: optimization 未產生 slicing_result.json (檢查 MASK_DIR 是否有圖?)"

    except Exception as e:
        return f"💥 {frame_path} 崩潰: {str(e)}"


def main():
    # 前置檢查 A: 確保輸出目錄存在
    os.makedirs(CONTROL_MAPS_DIR, exist_ok=True)
    os.makedirs(JSON_OUT_DIR, exist_ok=True)

    # 前置檢查 B: 確保有輸入剪影
    frame_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.png")))
    if not frame_files:
        print(f"❌ 錯誤：在 {INPUT_DIR} 找不到任何 PNG 檔案！請先執行 video_to_silhouettes.py")
        return

    # 前置檢查 C: 確保有素材遮罩 (ICAS 運算必備)
    mask_files = glob.glob(os.path.join(MASK_DIR, "*.png"))
    if not mask_files:
        print(f"⚠️ 警告：{MASK_DIR} 是空的！ICAS 無法進行拼貼優化。")
        print(f"💡 請先執行 generate_dummy_masks.py 來產生基礎遮罩。")
        return

    total_frames = len(frame_files)
    max_workers = 6

    print(f"🚀 啟動 9600X 多核心加速 (Workers: {max_workers})...")
    start_time = time.time()

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_frame, f) for f in frame_files]

        completed = 0
        for future in concurrent.futures.as_completed(futures):
            completed += 1
            # --- [關鍵修正：印出任務的實際回傳結果] ---
            result_msg = future.result()
            print(f"[{completed}/{total_frames}] {result_msg}")

    total_duration = time.time() - start_time
    print(f"\n✨ 處理結束！總耗時: {total_duration:.2f} 秒")

if __name__ == "__main__":
    main()