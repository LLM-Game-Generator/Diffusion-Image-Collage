from pathlib import Path
from src.utils.video_to_silhouettes import video_to_silhouettes
from src.utils.image_to_mask import images_to_masks_from_dir
from src.utils.images_to_video import VideoGenerator
from src.utils.batch_icas_tool import process_single_frame
from src.icas.dynamic_sas_generator import generate_dynamic_sequence
from src.icas.fast_batch_collage_renderer import batch_render_video_frames
from src.config import *


def remove_files_in_folder(folder: str):
    _folder = Path(folder)

    if _folder.exists() and _folder.is_dir():
        for file in _folder.glob('*'):
            if file.is_file():
                file.unlink()
    else:
        print("Path does not exist or it is not a folder")


class CollageSequenceGenerator(VideoGenerator):
    def get_image_paths(self, image_dir: str) -> list:
        files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')]
        return sorted(files)



def main(frame_no: str, remove_existing_collage_sequence: bool = True):
    if remove_existing_collage_sequence:
        remove_files_in_folder(JSON_SEQUENCE_DIR)
        remove_files_in_folder(COLLAGE_SEQUENCE_DIR)

    video_to_silhouettes(VIDEO_PATH, FRAME_DIR, TARGET_SIZE, FRAME_INTERVAL)

    frame_path = os.path.join(FRAME_DIR, f"{frame_no}.png")

    images_to_masks_from_dir(ASSETS_IMAGES_DIR, ASSETS_MASKS_DIR, overwrite=False)

    process_single_frame(frame_path, JSON_DIR, ASSETS_MASKS_DIR, ASSETS_IMAGES_DIR, False)

    generate_dynamic_sequence(
        frame_dir=FRAME_DIR,
        frame_no=frame_no,
        input_mask_folder=ASSETS_MASKS_DIR,
        output_dir=JSON_SEQUENCE_DIR,
        target_image="animal.jpg",
        total_frames=60,
        max_weight_multiplier=4.0
    )

    batch_render_video_frames(
        image_folder=ASSETS_IMAGES_DIR,
        json_sequence_dir=JSON_SEQUENCE_DIR,
        output_dir=COLLAGE_SEQUENCE_DIR,
        scaling_factor=2
    )

    generator = CollageSequenceGenerator()
    generator.create_video(COLLAGE_SEQUENCE_DIR, COLLAGE_SEQUENCE_VIDEO_PATH)


if __name__ == '__main__':
    main("frame_0003", remove_existing_collage_sequence=True)





