from rembg import remove
from PIL import Image
import os


def create_binary_mask(input_image_path, output_image_path):
    if os.path.exists(output_image_path):
        print("Output image already exists")
        return

    input_image = Image.open(input_image_path)

    output_mask = remove(input_image, only_mask=True)

    output_mask.save(output_image_path)
    print(f"轉換成功！遮罩已儲存為：{output_image_path}")


def images_to_masks_from_dir(input_dir, output_dir):
    if not os.path.exists(input_dir):
        return False

    os.makedirs(output_dir, exist_ok=True)

    images = os.listdir(input_dir)
    for file in images:
        image_path = os.path.join(input_dir, file)
        create_binary_mask(image_path, os.path.join(output_dir, file.replace(".jpg", ".png")))

    return True


if __name__ == "__main__":
    success = images_to_masks_from_dir("./assets/images", "./assets/mask")
    print(success)