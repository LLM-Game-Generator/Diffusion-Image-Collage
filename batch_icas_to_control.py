import os
import glob
import time
import concurrent.futures
from sas_optimization import optimization
from shape_decomposition import generate_cuts
from generate_control_maps import generate_control_maps

# 設定路徑
INPUT_DIR = "./input_silhouette_frames"
JSON_OUT_DIR = "./batch_output/jsons"
# MASK_DIR = "./resources/mask"
MASK_DIR = "./input_data/image_collections/children_mask"
CONTROL_MAPS_DIR = "./batch_output/control_maps"


def process_single_frame(frame_path):
    """
    單一影格的處理單元，將被多個進程同時調用
    """
    try:
        frame_name = os.path.basename(frame_path).split(".")[0]
        frame_json_dir = os.path.join(JSON_OUT_DIR, frame_name)

        if not os.path.exists(frame_json_dir):
            os.makedirs(frame_json_dir)

        generate_cuts(frame_path, frame_json_dir)
        # 1. 執行 ICAS 幾何優化 (CPU 密集)
        # quiet=True 關閉繪圖以節省時間
        optimization(frame_path, MASK_DIR, frame_json_dir, quiet=True)

        # 2. 產生 ControlNet 用的 Edge Map
        json_path = os.path.join(frame_json_dir, "slicing_result.json")
        if os.path.exists(json_path):
            target_filename = f"control_{frame_name}.png"
            generate_control_maps(
                json_path,
                CONTROL_MAPS_DIR,
                scaling_factor=1.0,
                mode='edge',
                output_filename=target_filename
            )
        return f"✅ {frame_name} 完成"
    except Exception as e:
        return f"❌ {frame_path} 失敗: {e}"


def main():
    if not os.path.exists(CONTROL_MAPS_DIR):
        os.makedirs(CONTROL_MAPS_DIR)

    frame_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.png")))
    total_frames = len(frame_files)

    # 根據 9600X 的核心數決定並行數量
    # 建議設為 6 (實體核心數) 或 10 (留一點給系統)
    max_workers = 6

    print(f"🚀 偵測到 AMD 9600X，啟動 {max_workers} 核心並行加速...")
    print(f"📦 總計影格數: {total_frames}")
    start_time = time.time()

    # 使用 ProcessPoolExecutor 進行多進程平行運算
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任務
        futures = [executor.submit(process_single_frame, f) for f in frame_files]

        # 追蹤進度
        completed = 0
        for future in concurrent.futures.as_completed(futures):
            completed += 1
            if completed % 10 == 0 or completed == total_frames:
                print(f" Progress: [{completed}/{total_frames}] ({(completed / total_frames) * 100:.1f}%)")

    end_time = time.time()
    total_duration = end_time - start_time
    print(f"\n✨ 全部處理完成！")
    print(f"⏱️ 總耗時: {total_duration:.2f} 秒")
    print(f"⚡ 平均速度: {total_frames / total_duration:.2f} FPS")


if __name__ == "__main__":
    main()