import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


VIDEO_PATH = os.path.join(BASE_DIR, "videos", "46012-448062061_medium.mp4")
FRAME_DIR = os.path.join(BASE_DIR, "input_silhouette_frames")

TARGET_SIZE = 512
FRAME_INTERVAL = 3


ASSETS_IMAGES_DIR = os.path.join(BASE_DIR, "assets", "images")
ASSETS_MASKS_DIR = os.path.join(BASE_DIR, "assets", "masks")


JSON_DIR = os.path.join(BASE_DIR, "batch_output", "jsons")
CONTROL_MAPS_DIR = os.path.join(BASE_DIR, "batch_output", "control_maps")
COLLAGE_VIDEO_PATH = os.path.join(BASE_DIR, "batch_output", "collage_video.mp4")


os.makedirs(FRAME_DIR, exist_ok=True)
os.makedirs(JSON_DIR, exist_ok=True)