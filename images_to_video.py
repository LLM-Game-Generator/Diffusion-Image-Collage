import cv2
import os


IMAGE_DIR = "./batch_output/jsons/"
VIDEO_PATH = './batch_output/collage_video.mp4'


if __name__ == '__main__':
    images = []
    frame_dir_list = os.listdir(IMAGE_DIR)
    for frame_dir in frame_dir_list:
        print(frame_dir)
        dir_path = os.path.join(IMAGE_DIR, frame_dir)
        files = os.listdir(dir_path)
        if "collage_white_space.png" in files:
            images.append(os.path.join(dir_path, "collage_white_space.png"))
    # 讀取第一張圖片來獲取長寬
    frame = cv2.imread(images[0])
    height, width, layers = frame.shape

    # 定義編碼器並建立 VideoWriter 物件
    # 'mp4v' 對應 .mp4 格式；24 為影格率 (fps)
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    video = cv2.VideoWriter(VIDEO_PATH, fourcc, 24, (width, height))

    for image in images:
        video.write(cv2.imread(image))

    cv2.destroyAllWindows()
    video.release()