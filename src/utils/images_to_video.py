import cv2
import os
import numpy as np
from src.config import JSON_DIR, COLLAGE_VIDEO_PATH


def images_to_video(image_dir: str, video_path: str):
    # 1. Collect and SORT the directories (to ensure frame_0001, frame_0002 order)
    frame_dir_list = sorted(os.listdir(image_dir))

    images = []
    for frame_dir in frame_dir_list:
        dir_path = os.path.join(image_dir, frame_dir)
        if not os.path.isdir(dir_path):
            continue

        img_path = os.path.join(dir_path, "collage_white_space.png")
        if os.path.exists(img_path):
            images.append(img_path)

    if not images:
        print("❌ No images found to process!")
        return

    # 2. Read first image to set base dimensions
    first_frame = cv2.imread(images[0])
    if first_frame is None:
        print(f"❌ Failed to read the first image: {images[0]}")
        return

    height, width, layers = first_frame.shape
    print(f"📺 Video Dimensions: {width}x{height} | Total Frames: {len(images)}")

    # 3. Ensure the output directory exists
    os.makedirs(os.path.dirname(video_path), exist_ok=True)

    # 4. Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(video_path, fourcc, 24, (width, height))

    for idx, img_path in enumerate(images):
        img = cv2.imread(img_path)

        if img is None:
            print(f"⚠️ Warning: Skipping frame {idx}, could not read {img_path}")
            continue

        # 5. RESIZE check (FFmpeg will fail if the size is different)
        if img.shape[0] != height or img.shape[1] != width:
            # Optionally resize it to fit the video dimensions
            img = cv2.resize(img, (width, height))

        video.write(img)
        if idx % 10 == 0:
            print(f"Processed {idx}/{len(images)} frames...")

    video.release()
    print(f"✅ Video successfully saved to: {video_path}")


if __name__ == '__main__':
    images_to_video(JSON_DIR, COLLAGE_VIDEO_PATH)