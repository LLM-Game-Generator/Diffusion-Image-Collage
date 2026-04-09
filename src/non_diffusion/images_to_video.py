import cv2
import os
from src.config import JSON_DIR, COLLAGE_VIDEO_PATH


def images_to_video(image_dir: str, video_path: str):
    images = []
    frame_dir_list = os.listdir(image_dir)
    for frame_dir in frame_dir_list:
        print(frame_dir)
        dir_path = os.path.join(image_dir, frame_dir)
        files = os.listdir(dir_path)
        if "collage_white_space.png" in files:
            images.append(os.path.join(dir_path, "collage_white_space.png"))
    # 讀取第一張圖片來獲取長寬
    frame = cv2.imread(images[0])
    height, width, layers = frame.shape

    # 定義編碼器並建立 VideoWriter 物件
    # 'mp4v' 對應 .mp4 格式；24 為影格率 (fps)
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    video = cv2.VideoWriter(video_path, fourcc, 24, (width, height))

    for image in images:
        video.write(cv2.imread(image))

    # cv2.destroyAllWindows()
    video.release()


if __name__ == '__main__':
    images_to_video(JSON_DIR, COLLAGE_VIDEO_PATH)