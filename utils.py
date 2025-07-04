import os
import cv2
import numpy as np
from super_gradients.training import models

def detection_voting(
    cam_folders,
    class_names= ['aunt_jemima_original_syrup', 'band_aid_clear_strips', 'bumblebee_albacore', 'cholula_chipotle_hot_sauce', 'crayola_24_crayons', 'hersheys_cocoa', 'honey_bunches_of_oats_honey_roasted', 'honey_bunches_of_oats_with_almonds', 'hunts_sauce', 'listerine_green', 'mahatma_rice', 'white_rain_body_wash', 'pringles_bbq', 'cheeze_it', 'hersheys_bar', 'redbull', 'mom_to_mom_sweet_potato_corn_apple', 'a1_steak_sauce', 'jif_creamy_peanut_butter', 'cinnamon_toast_crunch', 'arm_hammer_baking_soda', 'dr_pepper', 'haribo_gold_bears_gummi_candy', 'bulls_eye_bbq_sauce_original', 'reeses_pieces', 'clif_crunch_peanut_butter', 'mom_to_mom_butternut_squash_pear', 'pop_tararts_strawberry', 'quaker_big_chewy_chocolate_chip', 'spam', 'coffee_mate_french_vanilla', 'pepperidge_farm_milk_chocolate_macadamia_cookies', 'kitkat_king_size', 'snickers', 'toblerone_milk_chocolate', 'clif_z_bar_chocolate_chip', 'nature_valley_crunchy_oats_n_honey', 'ritz_crackers', 'palmolive_orange', 'crystal_hot_sauce', 'tapatio_hot_sauce', 'pepperidge_farm_milano_cookies_double_chocolate', 'nabisco_nilla_wafers', 'campbells_chicken_noodle_soup', 'frappuccino_coffee', 'chewy_dips_chocolate_chip', 'chewy_dips_peanut_butter', 'nature_vally_fruit_and_nut', 'cheerios', 'lindt_excellence_cocoa_dark_chocolate', 'hersheys_symphony', 'campbells_chunky_classic_chicken_noodle', 'martinellis_apple_juice', 'dove_pink', 'dove_white', 'david_sunflower_seeds', 'monster_energy', 'act_ii_butter_lovers_popcorn', 'coca_cola_glass_bottle', 'twix'],
    chkpoint_path ="/home/aistore51/checkpoint/yolo_nas_m_w_coco/RUN_20250627_153915_188556/ckpt_best.pth",
    output_txt_path="~/git/output/counts.txt",
    model_name="yolo_nas_s",
    num_classes=60,
):
    # 모델 로드
    model = models.get(
        model_name,
        num_classes=60,
        checkpoint_path=chkpoint_path
    )
    model.eval()

    num_cams = len(cam_folders)
    counts = np.zeros((num_classes, num_cams), dtype=int)

    # 캠별 예측
    for cam_idx, img_folder in enumerate(cam_folders):
        img_files = sorted([
            f for f in os.listdir(img_folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])
        for fname in img_files:
            img_path = os.path.join(img_folder, fname)
            pred = model.predict(img_path)
            labels = pred.prediction.labels.astype(int)
            for lbl in labels:
                counts[lbl, cam_idx] += 1

    # 다수결 기반 최종 카운트
    final_counts = []
    for row in counts:
        values, freqs = np.unique(row, return_counts=True)
        most_common_val = values[np.argmax(freqs)]
        final_counts.append(most_common_val)
    final_counts = np.array(final_counts)

    # 결과 저장
    with open(output_txt_path, "w") as f:
        for cls_id, count in enumerate(final_counts):
            class_name = class_names[cls_id]
            f.write(f"{cls_id:02d}\t{class_name}\t{count}\n")

    return final_counts
