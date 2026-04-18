import cv2
import os
import shutil
from abc import ABC, abstractmethod
from src.config import JSON_DIR, COLLAGE_VIDEO_PATH, COLLAGE_DIR


class VideoGenerator(ABC):
    """
    影片生成基底類別
    """

    def __init__(self, fps=24, codec='avc1'):
        self.fps = fps
        self.codec = codec

    @abstractmethod
    def get_image_paths(self, image_dir: str) -> list:
        """
        這是一個抽象方法，子類別必須實作它來決定如何獲取圖片清單。
        """
        pass

    def create_video(self, image_dir: str, video_path: str):
        """
        主要的主流程函數，使用者呼叫這個來產出影片。
        """
        # 1. 透過被 Override 的方法取得圖片列表
        images = self.get_image_paths(image_dir)

        if not images:
            print("❌ No images found to process!")
            return

        # 2. 讀取第一張圖設定維度
        first_frame = cv2.imread(images[0])
        if first_frame is None:
            print(f"❌ Failed to read the first image: {images[0]}")
            return

        height, width, _ = first_frame.shape
        print(f"📺 Video Dimensions: {width}x{height} | Total Frames: {len(images)}")

        # 3. 確保輸出目錄存在
        os.makedirs(os.path.dirname(video_path), exist_ok=True)

        # 4. 寫入影片
        fourcc = cv2.VideoWriter.fourcc(*self.codec)
        video = cv2.VideoWriter(video_path, fourcc, self.fps, (width, height))

        for idx, img_path in enumerate(images):
            img = cv2.imread(img_path)
            if img is None:
                print(f"⚠️ Warning: Skipping frame {idx}, could not read {img_path}")
                continue

            # 自動調整尺寸以防報錯
            if img.shape[0] != height or img.shape[1] != width:
                img = cv2.resize(img, (width, height))

            video.write(img)
            if idx % 10 == 0:
                print(f"Processed {idx}/{len(images)} frames...")

        video.release()
        print(f"✅ Video successfully saved to: {video_path}")


# --- 以下是具體的實作範例 ---

class CollageVideoGenerator(VideoGenerator):
    """
    這是你原本邏輯的實作：尋找子目錄下的 collage_white_space.png
    """

    def get_image_paths(self, image_dir: str) -> list:
        frame_dir_list = sorted(os.listdir(image_dir))
        image_paths = []

        for frame_dir in frame_dir_list:
            dir_path = os.path.join(image_dir, frame_dir)
            if not os.path.isdir(dir_path):
                continue

            img_path = os.path.join(dir_path, "collage.png")
            if os.path.exists(img_path):
                image_paths.append(img_path)

        return image_paths



if __name__ == '__main__':
    # 使用你原本的邏輯
    generator = CollageVideoGenerator(fps=24)
    generator.create_video(JSON_DIR, COLLAGE_VIDEO_PATH)