# main.py
import os
# os.environ["OMP_NUM_THREADS"]      = "1"
# os.environ["OPENBLAS_NUM_THREADS"]= "1"
# os.environ["MKL_NUM_THREADS"]     = "1"

import shutil
import glob
import csv
import itertools
import numpy as np
import postprocess_utils as pp
import smoothing_test

OUTPUT_DIR = "outputs"
RESULTS_DIR = "/home/aistore51/git/results"  # 캐시된 txt 파일들 위치

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

num_classes = len(CLASSES)
num_cams    = 5

def load_cached_results(results_dir):
    # 1) 파일 리스트 가져오기
    paths = sorted(glob.glob(os.path.join(results_dir, "event_*.txt")))
    T = len(paths)
    
    # 2) 미리 배열/리스트 할당
    event_ids   = np.zeros(T, dtype=int)
    raw_counts  = np.zeros((T, num_classes, num_cams), dtype=int)
    raw_boxes   = [None] * T

    # 3) 순회하면서 채워넣기
    for idx, p in enumerate(paths):
        fname = os.path.basename(p)             # "event_00001.txt"
        base, _ = os.path.splitext(fname)
        _, id_str = base.split('_')
        event_ids[idx] = int(id_str)

        # counts/boxes 초기화
        counts       = np.zeros((num_classes, num_cams), dtype=int)
        boxes_by_cam = {i: [] for i in range(num_cams)}

        with open(p, newline='') as f:
            reader = csv.reader(f, delimiter='\t')
            next(reader, None)
            # -- counts --
            for row in reader:
                if not row:
                    break
                cid = int(row[0])
                for cam_i, val in enumerate(row[1:]):
                    counts[cid, cam_i] = int(val)
            # -- boxes --
            next(reader, None)
            for row in reader:
                if len(row) < 4:
                    continue
                cam_i, cls_i, cx, cy = int(row[0]), int(row[1]), float(row[2]), float(row[3])
                boxes_by_cam[cam_i].append((cls_i, cx, cy))

        raw_counts[idx] = counts
        raw_boxes[idx]  = boxes_by_cam

    return event_ids, raw_counts, raw_boxes


def grid_search(event_ids, raw_counts, raw_boxes, combos, smoothing_params, image_size, cam_map):
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    T, C = raw_counts.shape[:2]
    print(f"[INFO] Total events: {T}, Classes: {C}")

    for cfg_idx, cfg in enumerate(combos):
        print(f"\n[INFO] Running config {cfg_idx}: {cfg}")
        cfg_dir = os.path.join(OUTPUT_DIR,f"cfg_{cfg_idx:03d}, idea:{cfg['idea']}")
        os.makedirs(cfg_dir, exist_ok=True)

        raw = []
        for t in range(T):
            counts       = raw_counts[t]
            boxes_by_cam = raw_boxes[t]

            try:
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
                    mask = pp.quadrant_presence_union(boxes_by_cam,
                                                      image_size,
                                                      cam_map,
                                                      min_hits=cfg["min_hits"], 
                                                      num_classes = 60)
                    res = (counts.max(axis=1) * mask).astype(int)
                elif cfg["idea"] == 5:
                    res = pp.quadrant_presence_max(boxes_by_cam,
                                                   counts,
                                                   image_size,
                                                   cam_map,
                                                   min_hits=cfg["min_hits"],
                                                   num_classes= 60)
                else:
                    raise ValueError(f"Unknown idea {cfg['idea']}")
            except Exception as e:
                print(f"[ERROR] At time {t}, config {cfg_idx} (idea {cfg['idea']}): {e}")
                raise

            raw.append(res)

        raw = np.stack(raw, axis=0)
        print(f"[INFO] Finished raw processing for config {cfg_idx}")

        for w, K in smoothing_params:
            print(f"    [INFO] Applying smoothing: window={w}, K={K}")
            out_dir = os.path.join(cfg_dir, f"win{w}_K{K}")
            os.makedirs(out_dir, exist_ok=True)

            try:
                sm = smoothing_test.temporal_mode_smoothing(raw, window=w)
                filt = smoothing_test.hysteresis_filter(sm, K) if K > 0 else sm
            except Exception as e:
                print(f"[ERROR] during smoothing/hysteresis: window={w}, K={K}: {e}")
                raise

            for t, eid in enumerate(event_ids):
                fn = os.path.join(out_dir, f"event_{eid:05d}.txt")
                try:
                    with open(fn, "w") as fw:
                        for cid, cnt in enumerate(filt[t]):
                            fw.write(f"{cid:02d}\t{CLASSES[cid]}\t{cnt}\n")
                except Exception as e:
                    print(f"[ERROR] while saving results for event {eid} at config {cfg_idx}, w={w}, K={K}: {e}")
                    raise

    print("[INFO] grid_search completed successfully.")


if __name__ == "__main__":
    # 0) 캐시된 txt에서 데이터 로드
    event_ids, raw_counts, raw_boxes = load_cached_results(RESULTS_DIR)

    # 1) 그리드 설정
    combos = []
    weights_list = [[1.0,0.9,0.8,0.6,0.6],[1.0,1.0,0.7,0.5,0.5]]
    for w,thr,mh in itertools.product(weights_list,[1.0,1.2],[1,2]):
        combos.append({"idea":1,"weights":w,"thr":thr,"min_hits":mh})
    for w,mh in itertools.product(weights_list,[1,2]):
        combos.append({"idea":2,"weights":w,           "min_hits":mh})
    combos.append({"idea":3,"pairs":[(0,4),(1,3)],   "min_hits":2})
    for mh in [1,2]:
        combos.append({"idea":4,"min_hits":mh})
        combos.append({"idea":5,"min_hits":mh})

    smoothing_params = list(itertools.product([3,5,7],[0,1,2]))

    # 2) 실행
    grid_search(event_ids, raw_counts, raw_boxes,
                combos, smoothing_params,
                image_size=(640,480),
                cam_map={0:3,1:4,2:2,3:0,4:1})