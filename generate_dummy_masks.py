import os
import cv2
import numpy as np


def create_dummy_masks(output_folder, count=20):
    """
    產生指定數量的虛擬遮罩圖片，用於 ICAS 演算法的幾何切割參考。
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"📁 已建立資料夾: {output_folder}")

    for i in range(count):
        # 隨機產生不同的長寬比，讓拼貼格子更多樣化
        width = np.random.randint(300, 600)
        height = np.random.randint(300, 600)

        # 建立純黑畫布
        mask = np.zeros((height, width), dtype=np.uint8)

        # 在中央畫一個白色的矩形（代表物體主體）
        margin = 20
        cv2.rectangle(mask, (margin, margin), (width - margin, height - margin), 255, -1)

        file_path = os.path.join(output_folder, f"dummy_{i:03d}.png")
        cv2.imwrite(file_path, mask)

    print(f"✅ 成功產生 {count} 張虛擬遮罩於 {output_folder}")
    print("💡 現在你可以重新執行 batch_icas_to_control.py 了！")


if __name__ == "__main__":
    # 指向你的 ICAS 腳本預期讀取的路徑
    target_path = "./resources/mask"
    create_dummy_masks(target_path, count=24)  # 產生 24 塊拼貼