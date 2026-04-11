from rembg import remove
from PIL import Image
from src.config import ASSETS_IMAGES_DIR, ASSETS_MASKS_DIR
import os


def create_binary_mask(input_image_path, output_image_path, overwrite=False):
    if os.path.exists(output_image_path):
        if not overwrite:
            print(f"檔案已存在，跳過：{output_image_path}")
            return
        else:
            print(f"檔案已存在，準備覆寫：{output_image_path}")

    try:
        input_image = Image.open(input_image_path)
        output_mask = remove(input_image, only_mask=True)
        output_mask.save(output_image_path)
        print(f"轉換成功！遮罩已儲存為：{output_image_path}")
    except Exception as e:
        print(f"處理檔案 {input_image_path} 時發生錯誤: {e}")


def images_to_masks_from_dir(input_dir, output_dir, overwrite=False):
    if not os.path.exists(input_dir):
        print(f"找不到輸入目錄：{input_dir}")
        return False

    os.makedirs(output_dir, exist_ok=True)

    images = os.listdir(input_dir)
    for file in images:
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
            image_path = os.path.join(input_dir, file)

            file_name_without_ext = os.path.splitext(file)[0]
            output_path = os.path.join(output_dir, f"{file_name_without_ext}.png")

            # print(os.path.exists(output_path))
            create_binary_mask(image_path, output_path, overwrite=overwrite)

    return True


if __name__ == "__main__":

    success = images_to_masks_from_dir(ASSETS_IMAGES_DIR, ASSETS_MASKS_DIR, overwrite=False)
    print(f"執行結果：{success}")