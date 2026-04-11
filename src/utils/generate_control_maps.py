import json
import numpy as np
import cv2
import os
import random


def generate_control_maps(json_path, output_dir, scaling_factor=1, mode='edge', output_filename=None):
    try:
        with open(json_path, 'r') as f:
            layout = json.load(f)
    except Exception as e:
        print(f"❌ 無法讀取 JSON: {e}")
        return

    h_raw, w_raw = layout['height'], layout['width']
    height = int(h_raw * scaling_factor)
    width = int(w_raw * scaling_factor)

    canvas = np.zeros((height, width, 3), np.uint8)

    if mode == 'seg':
        for i, part in enumerate(layout['parts']):
            pts = (np.array(part['coords']) * scaling_factor).astype(np.int32)

            random.seed(i)
            color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
            cv2.fillPoly(canvas, [pts], color)

        canvas = cv2.flip(canvas, 0)

    elif mode == 'edge':
        if layout.get('cuts'):
            cuts = np.array(layout['cuts']) * scaling_factor
            thickness = max(1, int(2 * scaling_factor))
            for cut in cuts:
                pt1 = tuple(cut[0].astype(int))
                pt2 = tuple(cut[1].astype(int))
                cv2.line(canvas, pt1, pt2, (255, 255, 255), thickness, cv2.LINE_AA)

        thickness_poly = max(1, int(scaling_factor))
        for part in layout['parts']:
            pts = (np.array(part['coords']) * scaling_factor).astype(np.int32)
            cv2.polylines(canvas, [pts], True, (255, 255, 255), thickness_poly)

        canvas = cv2.flip(canvas, 0)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    save_name = output_filename if output_filename else f"control_{mode}.png"
    save_path = os.path.join(output_dir, save_name)

    cv2.imwrite(save_path, canvas, [cv2.IMWRITE_PNG_COMPRESSION, 3])


if __name__ == '__main__':
    # 此處可加入 Multiprocessing 邏輯進行更高速的批次處理
    pass