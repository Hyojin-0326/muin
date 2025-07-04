# main.py
import os, shutil
import itertools
import numpy as np

from utils_yolo_nas import detection_voting  # 수정: boxes_by_cam까지 반환하도록
import postprocess_utils as pp
import utils_yolo_nas

OUTPUT_DIR = "outputs"
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

def grid_search(cam_base, combos, smoothing_params, image_size, cam_map):
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    os.makedirs(OUTPUT_DIR)

    # 이벤트 리스트 추출 (ref cam 폴더에 있는 jpg 기준)
    ref_folder = cam_base[2]
    imgs = sorted(os.glob(os.path.join(ref_folder, "*.jpg")))
    event_files = [os.path.basename(p) for p in imgs]
    event_ids   = [int(fn.split("_")[2].split(".")[0]) for fn in event_files]

    for cfg_idx, cfg in enumerate(combos):
        cfg_dir = os.path.join(OUTPUT_DIR, f"cfg_{cfg_idx:03d}")

        # 1) 먼저 각 이벤트별 raw 결과 수집
        raw = []  # list of shape (C,)
        for fn in event_files:
            # 각 카메라별 그 이벤트 이미지 경로
            cam_inputs = [os.path.join(cam, fn) for cam in cam_base]
            counts, boxes = utils_yolo_nas.detection(cam_inputs,
                                      chkpoint_path=cfg.get("chkpoint_path",None),
                                      model_name=cfg.get("model_name","yolo_nas_s"),
                                      num_classes=len(CLASSES),
                                      score_thres=cfg.get("score_thres",0.6),
                                      iou_thres=cfg.get("iou_thres",0.5),
                                      max_det=cfg.get("max_det",100))
            # 후처리 아이디어 적용
            if cfg["idea"] == 1:
                res = pp.weighted_vote_union(counts,
                                             np.array(cfg["weights"]),
                                             thr=cfg["thr"],
                                             min_hits=cfg["min_hits"])
            elif cfg["idea"] == 2:
                res = pp.weighted_union_simple(counts,
                                               np.array(cfg["weights"]),
                                               min_hits=cfg["min_hits"])
            elif cfg["idea"] == 3:
                res = pp.pairwise_presence_union(counts,
                                                 pairs=cfg["pairs"],
                                                 min_hits=cfg["min_hits"])
            elif cfg["idea"] == 4:
                mask = pp.quadrant_presence_union(boxes,
                                                  image_size,
                                                  cam_map,
                                                  min_hits=cfg["min_hits"])
                res  = (counts.max(axis=1) * mask).astype(int)
            elif cfg["idea"] == 5:
                res  = pp.quadrant_presence_max(boxes,
                                                counts,
                                                image_size,
                                                cam_map,
                                                min_hits=cfg["min_hits"])
            else:
                raise ValueError

            raw.append(res)

        raw_counts = np.stack(raw, axis=0)  # shape (T, C)

        # 2) 스무딩 파라미터 그리드 서치
        for w, K in smoothing_params:
            out_dir = os.path.join(cfg_dir, f"win{w}_K{K}")
            os.makedirs(out_dir, exist_ok=True)

            # 모드 스무딩
            sm = pp.temporal_mode_smoothing(raw_counts, window=w)
            # 히스테리시스
            if K > 0:
                filt = pp.hysteresis_filter(sm, K)
            else:
                filt = sm

            # 3) 각 이벤트별 결과 txt로 저장
            for idx, eid in enumerate(event_ids):
                with open(os.path.join(out_dir, f"event_{eid:05d}.txt"), "w") as f:
                    for cid, cnt in enumerate(filt[idx]):
                        f.write(f"{cid:02d}\t{CLASSES[cid]}\t{cnt}\n")

if __name__ == "__main__":
    cam_base = ["cam0/", "cam1/", "cam2/", "cam3/", "cam4/"]
    image_size = (640, 480)
    cam_map    = {0:3,1:4,2:2,3:0,4:1}

    # 카메라 후처리 그리드
    combos = []
    weights_list = [
        [1.0,0.9,0.8,0.6,0.6],
        [1.0,1.0,0.7,0.5,0.5],
    ]
    for w,thr,mh in itertools.product(weights_list,[1.0,1.2],[1,2]):
        combos.append({"idea":1, "weights":w, "thr":thr, "min_hits":mh})
    for w,mh in itertools.product(weights_list,[1,2]):
        combos.append({"idea":2, "weights":w,           "min_hits":mh})
    combos.append({"idea":3, "pairs":[(0,4),(1,3)],  "min_hits":2})
    for mh in [1,2]:
        combos.append({"idea":4, "min_hits":mh})
        combos.append({"idea":5, "min_hits":mh})

    # 스무딩 그리드: 윈도우 크기 × 히스테리시스 K (K=0이면 히스테리시스 스킵)
    smoothing_params = list(itertools.product([3,5,7], [0,1,2]))

    grid_search(cam_base, combos, smoothing_params, image_size, cam_map)