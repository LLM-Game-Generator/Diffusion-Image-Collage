# ICAS 動態影像拼貼管線指南 (Non-Diffusion 版)

本專案實作了高效能的自動化影像拼貼管線，專為快速產出動態拼貼視覺效果而設計。透過 `src/non_diffusion/main.py`，你可以一鍵完成從影片去背到拼貼合成的全流程。

## 📂 專案結構簡介

```text
.
├── src
│   ├── config.py                # 核心參數配置文件
│   ├── icas/                    # ICAS 幾何演算法 (MAD, SAS, Assembly)
│   ├── non_diffusion/
│   │   ├── main.py              # 自動化管線主程式
│   │   ├── video_to_silhouettes.py
│   │   ├── image_to_mask.py
│   │   └── images_to_video.py
│   └── assets/                  # 拼貼素材 (Images & Masks)
└── batch_output/                # 最終產出 (JSONs & Video)
```

## ⚙️ 配置 `src/config.py` 參數

在執行之前，請確保 `src/config.py` 內定義了以下關鍵變數：

| 參數名 | 說明 | 建議值 |
| :--- | :--- | :--- |
| `VIDEO_PATH` | 輸入影片的路徑 | `"src/videos/your_video.mp4"` |
| `FRAME_DIR` | 存放產生的剪影 PNG 資料夾 | `"src/input_silhouette_frames/"` |
| `TARGET_SIZE` | 處理解析度 (正方形) | `512` |
| `FRAME_INTERVAL` | 取樣間隔 (每 N 幀處理一次) | `2` 或 `3` (視流暢度需求) |
| `JSON_DIR` | 存放幾何優化結果的資料夾 | `"src/batch_output/jsons/"` |
| `ASSETS_IMAGES_PATH` | 拼貼素材原圖路徑 | `"src/assets/images/"` |
| `ASSETS_MASKS_PATH` | 素材對應的遮罩路徑 | `"src/assets/masks/"` |
| `COLLAGE_VIDEO_PATH` | 最終拼貼影片輸出路徑 | `"src/batch_output/collage_video.mp4"` |

## 🚀 如何執行 (Usage)

請確保你在專案根目錄下，並使用配置好的虛擬環境執行：

```powershell
python -m src.non_diffusion.main
```

### `main.py` 自動化流程說明：
1. **video_to_silhouettes**: 讀取 `VIDEO_PATH`，進行中心裁剪、去背並縮放至 `TARGET_SIZE`，存入 `FRAME_DIR`。
2. **images_to_masks_from_dir**: 檢查 `ASSETS_IMAGES_PATH`，若對應的 Mask 不存在，則自動在 `ASSETS_MASKS_PATH` 產生純白遮罩。
3. **multiprocessing**: 啟動多核心並行運算：
   - **SD**: 生成幾何切割線。
   - **SO**: 執行 SAS 優化分配素材（內含循環隊列補丁，素材再少也不崩潰）。
   - **CA**: 渲染拼貼畫（Scaling Factor = 2）。
4. **images_to_video**: 掃描 `JSON_DIR` 下的所有渲染圖，合成最終的 `collage_video.mp4`。

## 🛠️ 核心優化技術

### 多核心並行
主程式預設開啟 `max_workers=6`。ICAS 運算屬於 CPU 密集型任務，多進程處理能將數百幀的處理時間從數十分鐘壓縮至 **90-120 秒**。

### 循環素材分配
在 `sas_optimization.py` 中已套用循環隊列補丁，即使 `assets/images` 中只有 4 張圖，也能完美填滿擁有數百個格子的複雜剪影。

## ⚠️ 常見問題 (FAQ)

* **Q: 為什麼某些幀切出來只有一刀？**
    * A: 通常是因為剪影邊緣有雜訊或內部有空洞，導致幾何演算法提早終止。請檢查 `input_silhouette_frames` 中的圖片純淨度。
* **Q: 執行時報錯 `list index out of range`？**
    * A: 請確認 `config.py` 中的路徑結尾是否正確加上了分隔符號，或檢查 `ASSETS_MASKS_PATH` 是否真的有產生 PNG 檔案。


# Reference
* http://graphics.csie.ncku.edu.tw/shapedimagecollage/
* https://github.com/Dongee-W/Colla