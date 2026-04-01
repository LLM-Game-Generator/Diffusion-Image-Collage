import json
import requests
import websocket # pip install websocket-client
import uuid

# --- 設定區 ---
COMFYUI_URL = "127.0.0.1:8188" # 你的 ComfyUI 位址
WORKFLOW_FILE = r"D:/NCKU/CSIE-Project/comfyui.json" # 你導出的 API JSON 檔名
INPUT_IMAGE = "./output_dir/baby/control_edge.png"
CLIENT_ID = str(uuid.uuid4())

def upload_image(path):
    with open(path, 'rb') as f:
        files = {"image": f}
        data = {"overwrite": "true"}
        resp = requests.post(f"http://{COMFYUI_URL}/upload/image", files=files, data=data)
        return resp.json()

def queue_prompt(prompt):
    p = {"prompt": prompt, "client_id": CLIENT_ID}
    data = json.dumps(p).encode('utf-8')
    requests.post(f"http://{COMFYUI_URL}/prompt", data=data)

# --- 執行流程 ---
# 1. 上傳你的 ICAS 幾何控制圖
print(f"正在上傳控制圖...")
upload_result = upload_image(INPUT_IMAGE)
uploaded_filename = upload_result['name']

# 2. 讀取並修改工作流
with open(WORKFLOW_FILE, "r", encoding="utf-8") as f:
    workflow = json.load(f)

# 注意：你需要確認 Load Image 節點在 JSON 裡的編號 (例如 "10")
# 將它的輸入檔案名稱改為剛上傳的那張
# 可以透過文字搜尋 "image" 或 "filename" 找到對應位置
for node_id in workflow:
    if workflow[node_id]["_meta"]["title"] == "Load Image":
        workflow[node_id]["inputs"]["image"] = uploaded_filename
        print(f"已更新節點 {node_id} 的輸入圖為: {uploaded_filename}")

# 3. 發送給 ComfyUI 跑渲染
print("發送任務給 ComfyUI...")
queue_prompt(workflow)
print("任務已加入隊列！請去 ComfyUI 介面看結果。")