import os
import numpy as np
from super_gradients.training import models

# -----------------------------------
# 1) 클래스 리스트
# -----------------------------------
CLASSES = [
    'aunt_jemima_original_syrup', 'band_aid_clear_strips', 'bumblebee_albacore',
    'cholula_chipotle_hot_sauce', 'crayola_24_crayons', 'hersheys_cocoa',
    'honey_bunches_of_oats_honey_roasted', 'honey_bunches_of_oats_with_almonds',
    'hunts_sauce', 'listerine_green', 'mahatma_rice', 'white_rain_body_wash',
    'pringles_bbq', 'cheeze_it', 'hersheys_bar', 'redbull',
    'mom_to_mom_sweet_potato_corn_apple', 'a1_steak_sauce',
    'jif_creamy_peanut_butter', 'cinnamon_toast_crunch',
    'arm_hammer_baking_soda', 'dr_pepper', 'haribo_gold_bears_gummi_candy',
    'bulls_eye_bbq_sauce_original', 'reeses_pieces',
    'clif_crunch_peanut_butter', 'mom_to_mom_butternut_squash_pear',
    'pop_tararts_strawberry', 'quaker_big_chewy_chocolate_chip', 'spam',
    'coffee_mate_french_vanilla',
    'pepperidge_farm_milk_chocolate_macadamia_cookies', 'kitkat_king_size',
    'snickers', 'toblerone_milk_chocolate', 'clif_z_bar_chocolate_chip',
    'nature_valley_crunchy_oats_n_honey', 'ritz_crackers', 'palmolive_orange',
    'crystal_hot_sauce', 'tapatio_hot_sauce',
    'pepperidge_farm_milano_cookies_double_chocolate', 'nabisco_nilla_wafers',
    'campbells_chicken_noodle_soup', 'frappuccino_coffee',
    'chewy_dips_chocolate_chip', 'chewy_dips_peanut_butter',
    'nature_vally_fruit_and_nut', 'cheerios', 'lindt_excellence_cocoa_dark_chocolate',
    'hersheys_symphony', 'campbells_chunky_classic_chicken_noodle',
    'martinellis_apple_juice', 'dove_pink', 'dove_white',
    'david_sunflower_seeds', 'monster_energy', 'act_ii_butter_lovers_popcorn',
    'coca_cola_glass_bottle', 'twix'
]

# -----------------------------------
# 2) IoU 기반 클래스별 NMS 구현
# -----------------------------------
def nms(pred, score_thres=0.7, iou_thres=0.5, max_det=100):
    scores = pred.prediction.confidence  # shape (N,)
    boxes  = pred.prediction.bboxes_xyxy  # shape (N,4)
    labels = pred.prediction.labels.astype(int)  # shape (N,)
    idxs   = np.arange(len(scores))

    # 1) score threshold 필터링
    keep_mask = scores > score_thres
    idxs   = idxs[keep_mask]
    boxes  = boxes[keep_mask]
    scores = scores[keep_mask]
    labels = labels[keep_mask]

    final_keep = []

    # 2) 클래스별 NMS
    for cls in np.unique(labels):
        cls_mask  = labels == cls
        cls_idxs  = idxs[cls_mask]
        cls_boxes = boxes[cls_mask]
        cls_scores= scores[cls_mask]

        # score 내림차순 정렬
        order = np.argsort(-cls_scores)
        cls_idxs   = cls_idxs[order]
        cls_boxes  = cls_boxes[order]
        cls_scores = cls_scores[order]

        # NMS 루프
        while len(cls_idxs) > 0:
            # 최고 스코어 박스 선택
            i = cls_idxs[0]
            final_keep.append(i)
            if len(cls_idxs) == 1:
                break

            # IoU 계산
            box      = cls_boxes[0]
            others   = cls_boxes[1:]
            x1 = np.maximum(box[0], others[:,0])
            y1 = np.maximum(box[1], others[:,1])
            x2 = np.minimum(box[2], others[:,2])
            y2 = np.minimum(box[3], others[:,3])

            inter_w = np.clip(x2 - x1, 0, None)
            inter_h = np.clip(y2 - y1, 0, None)
            inter   = inter_w * inter_h

            area1 = (box[2]-box[0])*(box[3]-box[1])
            area2 = (others[:,2]-others[:,0])*(others[:,3]-others[:,1])
            union = area1 + area2 - inter
            iou   = inter / union

            # IoU ≤ threshold 유지
            keep_mask = iou <= iou_thres
            cls_idxs   = cls_idxs[1:][keep_mask]
            cls_boxes  = cls_boxes[1:][keep_mask]
            cls_scores = cls_scores[1:][keep_mask]

    # 3) 최종 top-max_det
    #    (원본 scores 배열에서 인덱스 기반으로 정렬)
    final_keep = sorted(final_keep, key=lambda i: scores[i], reverse=True)
    return np.array(final_keep[:max_det], dtype=int)

# -----------------------------------
# 3) Detection + Union + 파일출력
# -----------------------------------
def detection_voting(
    cam_folders,
    output_txt_path,
    class_names=CLASSES,
    chkpoint_path="/home/aistore51/checkpoint/yolo_nas_m_w_coco/RUN_20250627_153915_188556/ckpt_best.pth",
    model_name="yolo_nas_s",
    num_classes=60,
    score_thres=0.6,
    iou_thres=0.5,
    max_det=100
):
    # 모델 로딩
    model = models.get(
        model_name,
        num_classes=num_classes,
        checkpoint_path=chkpoint_path
    )
    model.eval()

    num_cams = len(cam_folders)
    counts   = np.zeros((num_classes, num_cams), dtype=int)

    # 각 cam 폴더별 inference + NMS + 카운팅
    for cam_idx, img_folder in enumerate(cam_folders):
        img_files = sorted([
            f for f in os.listdir(img_folder)
            if f.lower().endswith((".jpg",".jpeg",".png"))
        ])
        for fname in img_files:
            img_path = os.path.join(img_folder, fname)
            pred     = model.predict(img_path)

            # IoU-NMS 적용
            keep_idxs = nms(pred, score_thres, iou_thres, max_det)
            labels    = pred.prediction.labels.astype(int)[keep_idxs]

            # 클래스별 freq count
            unique, freqs = np.unique(labels, return_counts=True)
            for lbl, freq in zip(unique, freqs):
                counts[lbl, cam_idx] = freq

    # cams 간 union (max)으로 최종 카운트
    union_counts = np.max(counts, axis=1)

    # ────────────────────────────────────────────────────
    # 2대 이상 카메라에서 잡혀야만 카운팅 시작 (Boolean filter)
    # ────────────────────────────────────────────────────
    # 각 클래스별로 검출된 카메라 수 세기
    cam_hits = np.count_nonzero(counts > 0, axis=1)  # shape (num_classes,)
    # True인 애들만 카운팅 허용
    valid_mask = cam_hits >= 2                      # shape (num_classes,), dtype=bool

    # valid_mask False인 클래스들은 0으로 (noisy 제거)
    union_counts = union_counts * valid_mask.astype(int)
    # ────────────────────────────────────────────────────

    # 결과 파일 쓰기
    with open(output_txt_path, "w") as f:
        for cls_id, cnt in enumerate(union_counts):
            f.write(f"{cls_id:02d}\t{class_names[cls_id]}\t{cnt}\n")

    return union_counts