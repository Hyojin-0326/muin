import os
import cv2
import numpy as np
from super_gradients.training import models

CLASSES = ['aunt_jemima_original_syrup', 'band_aid_clear_strips', 'bumblebee_albacore', 'cholula_chipotle_hot_sauce', 'crayola_24_crayons', 'hersheys_cocoa', 'honey_bunches_of_oats_honey_roasted', 'honey_bunches_of_oats_with_almonds', 'hunts_sauce', 'listerine_green', 'mahatma_rice', 'white_rain_body_wash', 'pringles_bbq', 'cheeze_it', 'hersheys_bar', 'redbull', 'mom_to_mom_sweet_potato_corn_apple', 'a1_steak_sauce', 'jif_creamy_peanut_butter', 'cinnamon_toast_crunch', 'arm_hammer_baking_soda', 'dr_pepper', 'haribo_gold_bears_gummi_candy', 'bulls_eye_bbq_sauce_original', 'reeses_pieces', 'clif_crunch_peanut_butter', 'mom_to_mom_butternut_squash_pear', 'pop_tararts_strawberry', 'quaker_big_chewy_chocolate_chip', 'spam', 'coffee_mate_french_vanilla', 'pepperidge_farm_milk_chocolate_macadamia_cookies', 'kitkat_king_size', 'snickers', 'toblerone_milk_chocolate', 'clif_z_bar_chocolate_chip', 'nature_valley_crunchy_oats_n_honey', 'ritz_crackers', 'palmolive_orange', 'crystal_hot_sauce', 'tapatio_hot_sauce', 'pepperidge_farm_milano_cookies_double_chocolate', 'nabisco_nilla_wafers', 'campbells_chicken_noodle_soup', 'frappuccino_coffee', 'chewy_dips_chocolate_chip', 'chewy_dips_peanut_butter', 'nature_vally_fruit_and_nut', 'cheerios', 'lindt_excellence_cocoa_dark_chocolate', 'hersheys_symphony', 'campbells_chunky_classic_chicken_noodle', 'martinellis_apple_juice', 'dove_pink', 'dove_white', 'david_sunflower_seeds', 'monster_energy', 'act_ii_butter_lovers_popcorn', 'coca_cola_glass_bottle', 'twix']


def detection_voting(
    cam_folders,
    output_txt_path,
    class_names= CLASSES,
    chkpoint_path ="/home/aistore51/checkpoint/yolo_nas_m_w_coco/RUN_20250627_153915_188556/ckpt_best.pth",
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
            unique, freqs = np.unique(labels, return_counts=True)
            for lbl, freq in zip(unique, freqs):
                counts[lbl, cam_idx] = freq

    # %%
    # 합집합 방식으로 final_counts 계산 & 결과 내보내기 (수정된 부분 시작)
    # 1) 클래스별로 카운팅된 counts 행렬: shape (60, 5)
    #    이미 위에서 counts = np.zeros((60,5)) … 로 계산된 상태라고 가정

    # multiset union 이라면 cams 간의 최댓값을 각 클래스별로 취함
    union_counts = np.max(counts, axis=1)  # shape (60,)

    # 결과 저장
    with open(output_txt_path, "w") as f:
        for cls_id, count in enumerate(union_counts):
            class_name = class_names[cls_id]
            f.write(f"{cls_id:02d}\t{class_name}\t{count}\n")

    return union_counts
