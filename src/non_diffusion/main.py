import os
import glob
import time
import concurrent.futures
import src.icas.sas_optimization as so
import src.icas.shape_decomposition as sd
import src.icas.collage_assembly as ca
from src.non_diffusion.video_to_silhouettes import video_to_silhouettes
from src.non_diffusion.image_to_mask import images_to_masks_from_dir
from src.non_diffusion.images_to_video import images_to_video
from src.config import *


def process_single_frame(frame_path: str, mask_dir: str, assets_images_path: str):
    try:
        frame_name = os.path.basename(frame_path).split(".")[0]
        frame_json_dir = os.path.join(JSON_DIR, frame_name) + os.sep

        if not os.path.exists(frame_json_dir):
            os.makedirs(frame_json_dir)

        sd.generate_cuts(frame_path, frame_json_dir)
        so.optimization(frame_path, mask_dir, frame_json_dir, True)
        # Render collage
        ca.render_collage(assets_images_path, frame_json_dir, 2)


    except Exception as e:
        return f"💥 {frame_path} 崩潰: {str(e)}"


def multiprocessing(max_workers: int = 6):
    os.makedirs(JSON_DIR, exist_ok=True)

    frame_files = sorted(glob.glob(os.path.join(FRAME_DIR, "*.png")))
    if not frame_files:
        print(f"❌ 錯誤：在 {FRAME_DIR} 找不到任何 PNG 檔案！請先執行 video_to_silhouettes.py")
        return

    mask_files = glob.glob(os.path.join(ASSETS_MASKS_PATH, "*.png"))
    if not mask_files:
        print(f"⚠️ 警告：{ASSETS_MASKS_PATH} 是空的！ICAS 無法進行拼貼優化。")
        print(f"💡 請先執行 generate_dummy_masks.py 來產生基礎遮罩。")
        return

    total_frames = len(frame_files)

    print(f"🚀 啟動 9600X 多核心加速 (Workers: {max_workers})...")
    start_time = time.time()

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_single_frame, f, ASSETS_MASKS_PATH, ASSETS_IMAGES_PATH)
            for f in frame_files
        ]

        completed = 0
        for future in concurrent.futures.as_completed(futures):
            completed += 1
            result_msg = future.result()
            print(f"[{completed}/{total_frames}] {result_msg}")

    total_duration = time.time() - start_time
    print(f"\n✨ 處理結束！總耗時: {total_duration:.2f} 秒")




def main():
    video_to_silhouettes(VIDEO_PATH, FRAME_DIR, TARGET_SIZE, FRAME_INTERVAL)
    
    images_to_masks_from_dir(ASSETS_IMAGES_PATH, ASSETS_MASKS_PATH, overwrite=False)
    
    multiprocessing(max_workers=6)

    images_to_video(JSON_DIR, COLLAGE_VIDEO_PATH)




if __name__ == '__main__':
    main()





