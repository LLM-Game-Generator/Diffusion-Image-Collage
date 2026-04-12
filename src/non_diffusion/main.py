from src.utils.video_to_silhouettes import video_to_silhouettes
from src.utils.image_to_mask import images_to_masks_from_dir
from src.utils.images_to_video import CollageVideoGenerator
from src.utils.batch_icas_tool import frame_multiprocessing
from src.config import *



def main():
    video_to_silhouettes(VIDEO_PATH, FRAME_DIR, TARGET_SIZE, FRAME_INTERVAL)
    
    images_to_masks_from_dir(ASSETS_IMAGES_DIR, ASSETS_MASKS_DIR, overwrite=False)
    
    frame_multiprocessing(
        frame_dir=FRAME_DIR,
        json_dir=JSON_DIR,
        mask_dir=ASSETS_MASKS_DIR,
        assets_images_path=ASSETS_IMAGES_DIR,
        is_generate_control_maps=True,
        control_maps_dir=CONTROL_MAPS_DIR,
        max_workers=6
    )

    generator = CollageVideoGenerator()
    generator.create_video(JSON_DIR, COLLAGE_VIDEO_PATH)


if __name__ == '__main__':
    main()





