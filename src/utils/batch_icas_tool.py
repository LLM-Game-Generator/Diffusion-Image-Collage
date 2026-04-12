import os
import glob
import time
import traceback
import concurrent.futures
import src.icas.sas_optimization as so
import src.icas.shape_decomposition as sd
import src.icas.collage_assembly as ca
import src.icas.fast_image_collage as fca
from src.utils.generate_control_maps import generate_control_maps



def process_single_frame(frame_path: str, json_dir: str, mask_dir: str, assets_images_dir: str,
                         is_generate_control_maps: bool = False, control_maps_dir: str = None):
    try:
        frame_name = os.path.basename(frame_path).split(".")[0]
        frame_json_dir = os.path.join(json_dir, frame_name) + os.sep

        if not os.path.exists(frame_json_dir):
            os.makedirs(frame_json_dir)

        sd.generate_cuts(frame_path, frame_json_dir)

        so.optimization(frame_path, mask_dir, frame_json_dir, True)

        # Render collage
        # ca.render_collage(assets_images_dir, frame_json_dir, 1)
        fca.fast_render_collage(assets_images_dir, frame_json_dir, 2)

        # Generate control map
        if is_generate_control_maps and control_maps_dir is not None:
            json_path = os.path.join(frame_json_dir, "slicing_result.json")
            if os.path.exists(json_path):
                target_filename = f"control_{frame_name}.png"
                generate_control_maps(
                    json_path,
                    control_maps_dir,
                    scaling_factor=1.0,
                    mode='edge',
                    output_filename=target_filename
                )
                return f"✅ {frame_name} 完成"
            else:
                return f"❌ {frame_name} 失敗: optimization 未產生 slicing_result.json (檢查 MASK_DIR 是否有圖?)"

        return f"✅ {frame_name} 完成"

    except Exception as e:
        error_stack = traceback.format_exc()
        print(f"\n--- ERROR IN {frame_path} ---\n{error_stack}\n----------------------------\n")
        return f"💥 {frame_path} 崩潰: {str(e)}"


def frame_multiprocessing(frame_dir: str, json_dir: str, mask_dir: str, assets_images_path: str,
                          is_generate_control_maps: bool = False, control_maps_dir: str = None, max_workers: int = 6):
    os.makedirs(json_dir, exist_ok=True)

    frame_files = sorted(glob.glob(os.path.join(frame_dir, "*.png")))
    if not frame_files:
        print(f"❌ 錯誤：在 {frame_dir} 找不到任何 PNG 檔案！請先執行 video_to_silhouettes.py")
        return

    mask_files = glob.glob(os.path.join(mask_dir, "*.png"))
    if not mask_files:
        print(f"⚠️ 警告：{mask_dir} 是空的！ICAS 無法進行拼貼優化。")
        print(f"💡 請先執行 generate_dummy_masks.py 來產生基礎遮罩。")
        return

    total_frames = len(frame_files)

    if is_generate_control_maps and control_maps_dir is not None:
        os.makedirs(control_maps_dir, exist_ok=True)

    print(f"多核心加速 (Workers: {max_workers})...")
    start_time = time.time()

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(process_single_frame, f, json_dir, mask_dir, assets_images_path, is_generate_control_maps,
                            control_maps_dir)
            for f in frame_files
        ]

        completed = 0
        for future in concurrent.futures.as_completed(futures):
            completed += 1
            result_msg = future.result()
            print(f"[{completed}/{total_frames}] {result_msg}")

    total_duration = time.time() - start_time
    print(f"\n 處理結束！總耗時: {total_duration:.2f} 秒")
