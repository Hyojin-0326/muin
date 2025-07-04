import os
import cv2
import numpy as np
from super_gradients.training import models

CLASSES = ['aunt_jemima_original_syrup', 'band_aid_clear_strips', 'bumblebee_albacore', 'cholula_chipotle_hot_sauce', 'crayola_24_crayons', 'hersheys_cocoa', 'honey_bunches_of_oats_honey_roasted', 'honey_bunches_of_oats_with_almonds', 'hunts_sauce', 'listerine_green', 'mahatma_rice', 'white_rain_body_wash', 'pringles_bbq', 'cheeze_it', 'hersheys_bar', 'redbull', 'mom_to_mom_sweet_potato_corn_apple', 'a1_steak_sauce', 'jif_creamy_peanut_butter', 'cinnamon_toast_crunch', 'arm_hammer_baking_soda', 'dr_pepper', 'haribo_gold_bears_gummi_candy', 'bulls_eye_bbq_sauce_original', 'reeses_pieces', 'clif_crunch_peanut_butter', 'mom_to_mom_butternut_squash_pear', 'pop_tararts_strawberry', 'quaker_big_chewy_chocolate_chip', 'spam', 'coffee_mate_french_vanilla', 'pepperidge_farm_milk_chocolate_macadamia_cookies', 'kitkat_king_size', 'snickers', 'toblerone_milk_chocolate', 'clif_z_bar_chocolate_chip', 'nature_valley_crunchy_oats_n_honey', 'ritz_crackers', 'palmolive_orange', 'crystal_hot_sauce', 'tapatio_hot_sauce', 'pepperidge_farm_milano_cookies_double_chocolate', 'nabisco_nilla_wafers', 'campbells_chicken_noodle_soup', 'frappuccino_coffee', 'chewy_dips_chocolate_chip', 'chewy_dips_peanut_butter', 'nature_vally_fruit_and_nut', 'cheerios', 'lindt_excellence_cocoa_dark_chocolate', 'hersheys_symphony', 'campbells_chunky_classic_chicken_noodle', 'martinellis_apple_juice', 'dove_pink', 'dove_white', 'david_sunflower_seeds', 'monster_energy', 'act_ii_butter_lovers_popcorn', 'coca_cola_glass_bottle', 'twix']

# %%
model = models.get(
    "yolo_nas_s",
    num_classes=60,
    checkpoint_path="/home/aistore51/checkpoint/yolo_nas_m_w_coco/RUN_20250604_165358_737109/ckpt_best.pth"
)
model.eval()

# %%
cam_folders = [
    f"/home/aistore51/Datasets/4.testset_sample/cam{i}/" for i in range(5)
]
num_classes = 60
num_cams = len(cam_folders)


# %%
# 디텍션 캠별로 돌림 (수정된 부분 시작)
counts = np.zeros((num_classes, num_cams), dtype=int)

for cam_idx, img_folder in enumerate(cam_folders):
    # 이번에는 모든 이미지가 아니라, 각 캠당 testset_event_10029_{i}.jpg 하나만
    img_name = f"testset_event_10023_{cam_idx}.jpg"
    img_path = os.path.join(img_folder, img_name)

    print(f"[DEBUG] ===== Processing CAM {cam_idx} =====")
    print(f"[DEBUG] Image path: {img_path}")

    # 실제 디텍션
    pred = model.predict(img_path)
    labels = pred.prediction.labels.astype(int)

    # 캠별로 라벨 카운팅
    unique, freqs = np.unique(labels, return_counts=True)
    for lbl, freq in zip(unique, freqs):
        counts[lbl, cam_idx] = freq

    print(f"[DEBUG] counts[:, {cam_idx}]: {counts[:, cam_idx]}\n")

# 전체 카운팅 행렬 한 번에 보여줌
print("[DEBUG] 최종 counts matrix (60×5):")
print(counts)
# 디텍션 캠별로 돌림 (수정된 부분 끝)

# %%
classes = CLASSES
# 합집합 방식으로 final_counts 계산 & 결과 내보내기 (수정된 부분 시작)
# 1) 클래스별로 카운팅된 counts 행렬: shape (60, 5)
#    이미 위에서 counts = np.zeros((60,5)) … 로 계산된 상태라고 가정

# multiset union 이라면 cams 간의 최댓값을 각 클래스별로 취함
union_counts = np.max(counts, axis=1)  # shape (60,)

# 2) 라벨 텍스트로 저장
with open("final_union_counts_labeled.txt", "w") as f:
    for cls_id, count in enumerate(union_counts):
        class_name = classes[cls_id]
        f.write(f"{cls_id:02d}\t{class_name}\t{count}\n")

print("[DEBUG] ▶ 합집합 결과 저장됨: final_union_counts_labeled.txt")

# 3) 원본 counts 행렬(60×5)도 텍스트로 출력
np.savetxt("counts_matrix.txt", counts, fmt="%d", delimiter="\t")
print("[DEBUG] ▶ counts matrix 저장됨: counts_matrix.txt")
print("[DEBUG] 최종 union_counts 배열:\n", union_counts)
print("[DEBUG] counts matrix:\n", counts)
# 합집합 방식으로 final_counts 계산 & 결과 내보내기 (수정된 부분 끝)
