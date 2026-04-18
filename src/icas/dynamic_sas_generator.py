import json
import os
import sys
import copy
from src.config import *
from shapely.geometry import LineString, Point
from shapely import ops

import src.icas.shape_decomposition as sd
from src.icas.sas_optimization import (
    Partition, build_medial_graph, find_center, patch_to_center, load_mask,
    extract_foreground, process_image_for_optimization, forest_initialization,
    calculate_image_assignment, assign_image, forest_optimization,
    extract_forest_geometry, min_rec, centroid_cut
)


def get_node_weight(node, multipliers):
    if node.is_leaf():
        img_id = node.assignment['id']
        return node.base_area * multipliers.get(img_id, 1.0)
    else:
        return get_node_weight(node.left_child, multipliers) + get_node_weight(node.right_child, multipliers)


def match_polygons(new_pieces, old_left, old_right):
    iou_0_left = new_pieces[0].intersection(old_left).area
    iou_1_left = new_pieces[1].intersection(old_left).area

    if iou_0_left > iou_1_left:
        return new_pieces[0], new_pieces[1]
    else:
        return new_pieces[1], new_pieces[0]


def record_baseline(node):
    if node.is_leaf():
        node.base_area = node.polygon.area
    else:
        record_baseline(node.left_child)
        record_baseline(node.right_child)
        node.base_area = node.left_child.base_area + node.right_child.base_area

        try:
            cut_geom = LineString(node.cut)
            node.cut_center = cut_geom.centroid
        except:
            node.cut_center = node.polygon.centroid


def weighted_temporal_optimization(tree_node, medial_axis, multipliers):
    if tree_node.is_leaf():
        convex = tree_node.polygon.convex_hull.simplify(10)
        optimal = min_rec(convex.exterior.coords, tree_node.assignment['aspect_ratio'],
                          list(convex.representative_point().coords)[0])
        tree_node.assignment["coord"] = optimal[0]
        return

    w_left = get_node_weight(tree_node.left_child, multipliers)
    w_right = get_node_weight(tree_node.right_child, multipliers)

    base_f_left = tree_node.left_child.base_area / tree_node.base_area if tree_node.base_area > 0 else 0.5

    total_w = w_left + w_right
    target_f_left = w_left / total_w if total_w > 0 else 0.5

    c_left = tree_node.left_child.polygon.centroid
    c_right = tree_node.right_child.polygon.centroid
    cut_center = tree_node.cut_center

    shift_ratio = target_f_left - base_f_left

    if shift_ratio > 0:
        amount = shift_ratio / (1.0 - base_f_left) if base_f_left < 1 else 0
        shifted_c_x = cut_center.x + amount * (c_right.x - cut_center.x)
        shifted_c_y = cut_center.y + amount * (c_right.y - cut_center.y)
    else:
        amount = -shift_ratio / base_f_left if base_f_left > 0 else 0
        shifted_c_x = cut_center.x + amount * (c_left.x - cut_center.x)
        shifted_c_y = cut_center.y + amount * (c_left.y - cut_center.y)

    shifted_c = (shifted_c_x, shifted_c_y)

    is_axial = (tree_node.configuration in [0, 1])
    cut_dir = tree_node.get_axial(medial_axis) if is_axial else tree_node.get_crosswise(medial_axis)

    cut_len = max(tree_node.get_size(cut_dir)[0], tree_node.get_size(cut_dir)[1]) * 3
    cut_line = LineString([Point(p) for p in centroid_cut(shifted_c, cut_dir, cut_len)])

    split_result = ops.split(tree_node.polygon, cut_line)
    pieces = list(split_result.geoms)

    if len(pieces) >= 2:
        pieces.sort(key=lambda x: -x.area)
        new_left, new_right = match_polygons(pieces[:2], tree_node.left_child.polygon, tree_node.right_child.polygon)
        tree_node.left_child.polygon = new_left
        tree_node.right_child.polygon = new_right

        cut_geom = tree_node.polygon.intersection(cut_line)
        if cut_geom.geom_type == 'MultiLineString':
            lines = list(cut_geom.geoms)
            lines.sort(key=lambda x: -x.length)
            cut_geom = lines[0]
        tree_node.cut = list(cut_geom.coords)

    weighted_temporal_optimization(tree_node.left_child, medial_axis, multipliers)
    weighted_temporal_optimization(tree_node.right_child, medial_axis, multipliers)


def generate_dynamic_sequence(
        frame_dir: str,
        frame_no: str,
        input_mask_folder: str,
        output_dir: str,
        target_images_config: dict = None,
        total_frames: int = None,
):
    if target_images_config is None:
        target_images_config = {f.replace(".png", ".jpg"): 1.0 for f in os.listdir(input_mask_folder) if f.endswith('.png')}

    if total_frames is None:
        total_frames = 60

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    import cv2
    print("🚀 正在初始化第一幀 (Frame 0) 排版骨架...")

    frame_path = os.path.join(frame_dir, f"{frame_no}.png")
    image = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
    polygon = sd.generate_canvas_polygon(image)[0]

    final_cut_json_path = os.path.join(os.path.dirname(output_dir), "jsons", frame_no, "final_cut.json")

    with open(final_cut_json_path) as f:
        prediction = json.load(f)

    prediction_partition = Partition(polygon)
    for cut in prediction:
        prediction_partition.add_cut(cut)

    medial_interior_input = sd.prepare_for_medial_axis(image, complement=False)
    ma_int = sd.ridge_medial_axis(medial_interior_input, ridge_threshold=0.39, small_threshold=5)
    multilinestring_int = sd.build_medial_multilinestring(ma_int[0])
    final_medial_vertices_int = sd.redistribute_vertices(multilinestring_int[0], 5)

    convex_parts = prediction_partition.list_leaves()
    G = build_medial_graph(final_medial_vertices_int, multilinestring_int[1], ma_int[1])
    boundary_vertices = sd.redistribute_vertices(LineString(polygon.exterior.coords), 5)
    center_id = find_center(polygon, G, boundary_vertices)
    convex_parts.sort(key=lambda x: -1 * patch_to_center(x, center_id, G))

    image_ids = [f.split(".")[0] for f in os.listdir(input_mask_folder) if f.endswith('.png')]
    image_dict = []
    image_template = {"filename": "", "foreground_exists": True, "foreground": [], "assigned_part": 0}

    for image_id in image_ids:
        label = load_mask(os.path.join(input_mask_folder, image_id + ".png"))
        x1, x2, y1, y2, foreground_exist = extract_foreground(label)
        it = image_template.copy()
        it["filename"] = image_id + ".jpg"  # 對齊真實檔名
        it["foreground"] = [x1, x2, y1, y2]
        if not foreground_exist:
            it['foreground_exists'] = False
        image_dict.append(it)

    images = process_image_for_optimization(image_dict)

    # 批次防呆鎖定機制：精準尋找多個目標圖片
    target_img_ids = {}  # img_id -> target_multiplier
    available_filenames = [d['filename'] for d in image_dict]

    for target_filename, target_multiplier in target_images_config.items():
        found = False
        for img in images:
            img_filename = image_dict[img['id']]['filename']
            if target_filename.lower() in img_filename.lower():
                target_img_ids[img['id']] = target_multiplier
                found = True
                print(f"✅ 成功鎖定目標：{img_filename} (目標倍率: {target_multiplier}x)")
                break
        if not found:
            print(f"⚠️ [警告] 找不到設定檔中的圖片 '{target_filename}'，將自動忽略此項！")

    if not target_img_ids:
        print(f"\n❌ [嚴重錯誤] 設定檔中所有的圖片都找不到！")
        print(f"💡 目前系統載入的圖片清單有: {available_filenames}")
        sys.exit(1)

    ss = forest_initialization(convex_parts, len(images), prediction_partition.root.polygon.area, True,
                               multilinestring_int)
    kk = calculate_image_assignment(images, ss[1])
    assign_image(ss[0], kk)

    result = forest_optimization(ss[0], multilinestring_int)
    forest = [r[1] for r in result]
    geometry = extract_forest_geometry(forest)

    for tree in forest:
        record_baseline(tree)

    inv_map = {v: k for k, v in geometry[1].items()}
    for k in inv_map:
        image_dict[k]['assigned_part'] = inv_map[k]

    base_optimization_output = {
        "images": image_dict,
        "parts": geometry[0],
        "width": image.shape[1],
        "height": image.shape[0],
        "cuts": geometry[2] + prediction
    }

    with open(os.path.join(output_dir, 'slicing_0000.json'), 'w') as f:
        json.dump(base_optimization_output, f)
    print("  [Generated] slicing_0000.json (基準佈局)")

    # 開始平滑推擠 (多目標線性插值)
    for frame_idx in range(1, total_frames):
        multipliers = {}
        for img in images:
            img_id = img['id']
            if img_id in target_img_ids:
                target_mult = target_img_ids[img_id]
                # 依時間比例計算目前的倍率 (從 1.0 漸變到 target_mult)
                current_mult = 1.0 + (target_mult - 1.0) * (frame_idx / (total_frames - 1))
                multipliers[img_id] = current_mult
            else:
                multipliers[img_id] = 1.0

        for tree in forest:
            weighted_temporal_optimization(tree, multilinestring_int[0], multipliers)

        new_geometry = extract_forest_geometry(forest)

        current_output = copy.deepcopy(base_optimization_output)
        current_output["parts"] = new_geometry[0]
        current_output["cuts"] = new_geometry[2] + prediction

        out_filename = f"slicing_{frame_idx:04d}.json"
        with open(os.path.join(output_dir, out_filename), 'w') as f:
            json.dump(current_output, f)

        print(f"  [Generated] {out_filename}")

    print(f"✅ 成功生成 {total_frames} 幀多目標平滑推擠序列！")


if __name__ == "__main__":
    # 在這裡設定多個目標！大於 1 放大，小於 1 縮小。
    targets = {
        "animal.jpg": 4.0,  # 第一張圖片放大 4 倍
        "flower.jpg": 0.5  # 假設你有這張圖片，它會縮小一半
    }

    generate_dynamic_sequence(
        frame_dir=FRAME_DIR,
        frame_no="frame_0000",
        input_mask_folder=ASSETS_MASKS_DIR,
        output_dir=JSON_SEQUENCE_DIR,
        target_images_config=targets,
        total_frames=60,
    )