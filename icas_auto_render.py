import os
import time
import json
import requests
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# --- 設定區 ---
# 監控 ICAS 產出線稿的資料夾
WATCH_DIR = r"D:\NCKU\CSIE-Project\Colla\batch_output\control_maps"
# ComfyUI API 位址
COMFYUI_URL = "http://127.0.0.1:8188"
# 之前存好的 API 工作流 JSON 檔案
WORKFLOW_JSON_PATH = "dynamic_collage_api.json"
# 觸發條件：當資料夾內達到多少張圖片時開始渲染
FRAME_THRESHOLD = 16


class ICASHandler(FileSystemEventHandler):
    def __init__(self):
        self.is_processing = False

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(".png"):
            self.check_and_trigger()

    def check_and_trigger(self, quiet=False):
        if self.is_processing: return
        files = [f for f in os.listdir(WATCH_DIR) if f.endswith(".png")]
        count = len(files)

        if count >= FRAME_THRESHOLD:
            print(f"[{time.strftime('%H:%M:%S')}] 🚀 準備渲染 {count} 張影格...")
            self.is_processing = True
            try:
                # 確保檔案已完全寫入
                time.sleep(2)
                self.send_to_comfyui(count)
                print(f"[{time.strftime('%H:%M:%S')}] ✅ 請求成功發送。請觀察 ComfyUI 視窗進度。")
            except Exception as e:
                print(f"[{time.strftime('%H:%M:%S')}] ❌ 失敗: {e}")
            finally:
                # 進入冷卻期
                time.sleep(30)
                self.is_processing = False
        else:
            if not quiet:
                print(f"[{time.strftime('%H:%M:%S')}] 進度: {count}/{FRAME_THRESHOLD}")

    def send_to_comfyui(self, frame_count):
        if not os.path.exists(WORKFLOW_JSON_PATH):
            raise FileNotFoundError(f"找不到工作流 JSON: {WORKFLOW_JSON_PATH}")

        with open(WORKFLOW_JSON_PATH, "r", encoding="utf-8") as f:
            workflow = json.load(f)

        safe_dir = WATCH_DIR.replace("\\", "/")

        # --- [資工系修復：針對長影片滑動窗口的最終對齊] ---
        for node_id, node in workflow.items():
            ctype = node.get("class_type")

            # 1. 對齊影格讀取 (Node 3)
            if ctype == "VHS_LoadImages":
                node["inputs"]["directory"] = safe_dir
                # node["inputs"]["image_load_cap"] = frame_count
                node["inputs"]["path_filter"] = "*.png"

            # 2. 對齊 Latent 數量 (Node 9)
            # if ctype == "EmptyLatentImage":
            #     node["inputs"]["batch_size"] = frame_count

            # 3. 補全影片合成參數 (Node 10)
            if ctype == "VHS_VideoCombine":
                node["inputs"]["filename_prefix"] = f"ICAS_Final_{int(time.time())}"
                node["inputs"]["format"] = "video/h264-mp4"
                node["inputs"]["save_output"] = True

            # 4. --- [核心修復：修正 ControlNet 以支援滑動窗口] ---
            if ctype == "ControlNetApplyAdvanced":
                # A. 確保強度與百分比存在
                node["inputs"]["strength"] = 1.0
                node["inputs"]["start_percent"] = 0.0
                node["inputs"]["end_percent"] = 1.0
                # B. 強制連接 VAE (來自 CheckpointLoader 節點 1 的輸出索引 2)
                # 這是解決 "ControlNet may not support sliding window" 的關鍵，因為它需要 VAE 進行編碼
                node["inputs"]["vae"] = ["1", 2]
                print(f"已強制對齊進階 ControlNet 節點 {node_id} 的 VAE 與百分比參數。")

        # 發送至 ComfyUI
        response = requests.post(f"{COMFYUI_URL}/prompt", json={"prompt": workflow})
        if response.status_code != 200:
            error_data = response.json()
            formatted_error = json.dumps(error_data, indent=2, ensure_ascii=False)
            raise Exception(f"ComfyUI 報錯:\n{formatted_error}")


if __name__ == "__main__":
    event_handler = ICASHandler()
    observer = Observer()
    observer.schedule(event_handler, WATCH_DIR, recursive=False)
    observer.start()

    print("🎨 ICAS 穩定版自動渲染服務啟動...")
    event_handler.check_and_trigger()

    try:
        while True:
            time.sleep(15)
            event_handler.check_and_trigger(quiet=True)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()