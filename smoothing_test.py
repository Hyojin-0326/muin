import os
import cv2
import numpy as np
import glob
import utils_yolo_nas

# — 함수 정의 —————————————————————————————————————————————

def temporal_mode_smoothing(raw_counts: np.ndarray, window: int) -> np.ndarray:
    """
    인접 프레임 다수결(mode) 스무딩
    raw_counts: shape (T, C)
    window: 윈도우 크기 (홀수 권장)
    return smoothed_counts: shape (T, C)
    """
    pad = window // 2
    padded = np.pad(raw_counts, ((pad, pad), (0, 0)), mode='edge')
    T, C = raw_counts.shape
    smoothed = np.zeros_like(raw_counts)
    for t in range(T):
        wblock = padded[t : t + window]  # shape (window, C)
        # 클래스별 최빈값 계산
        mode_vals = []
        for c in range(C):
            vals, freqs = np.unique(wblock[:, c], return_counts=True)
            mode_vals.append(vals[np.argmax(freqs)])
        smoothed[t] = mode_vals
    return smoothed

def hysteresis_filter(smoothed: np.ndarray, K: int) -> np.ndarray:
    """
    연속 K프레임 유지해야만 실제 state로 반영(히스테리시스)
    smoothed: shape (T, C)
    K: 유지 프레임 수
    return filtered: shape (T, C)
    """
    T, C = smoothed.shape
    filtered = np.zeros_like(smoothed)
    # 초기 상태
    current = smoothed[0].copy()
    filtered[0] = current
    stable = np.zeros(C, dtype=int)

    for t in range(1, T):
        for c in range(C):
            if smoothed[t, c] == current[c]:
                stable[c] += 1
            else:
                stable[c] = 1
            # K프레임 연속 유지됐을 때만 state 변경 허용
            if stable[c] >= K:
                current[c] = smoothed[t, c]
        filtered[t] = current
    return filtered

# — 메인 스크립트 ———————————————————————————————————————————

# 설정값



# txt_dir  = "/home/aistore51/git/output"
# img_dir  = "/home/aistore51/Datasets/4.testset_sample/cam2"
# out_base = "/home/aistore51/git/out_images"
# os.makedirs(out_base, exist_ok=True)

# CLASSES     = utils_yolo_nas.CLASSES
# num_classes = len(CLASSES)

# # raw_counts 읽기
# txt_files = sorted(glob.glob(os.path.join(txt_dir, "event_*.txt")))
# event_ids = [int(os.path.basename(p).split("_")[1].split(".")[0]) for p in txt_files]
# raw_counts = np.zeros((len(event_ids), num_classes), dtype=int)
# for i, path in enumerate(txt_files):
#     with open(path) as f:
#         for line in f:
#             cid, _, cnt = line.strip().split("\t")
#             raw_counts[i, int(cid)] = int(cnt)


# # 스무딩 윈도우 & 히스테리시스 파라미터
# smoothing_windows = [3, 5, 7]
# K_values = [0, 1, 2]  # 여러 K값 테스트

# for w in smoothing_windows:
#     smoothed = temporal_mode_smoothing(raw_counts, w)
    
#     for k in K_values:
#         out_dir = os.path.join(out_base, f"win{w}_K{k}")
#         os.makedirs(out_dir, exist_ok=True)

#         # 2) 히스테리시스 필터(k=0이면 건너뜀)
#         if k > 0:
#             filtered = hysteresis_filter(smoothed, k)
#         else:
#             filtered = smoothed.copy()

#         # 변화 감지 & 오버레이
#         prev = filtered[0]
#         for idx, eid in enumerate(event_ids):
#             curr = filtered[idx]
#             diff = curr - prev

#             # change 텍스트
#             changes = []
#             for c in range(num_classes):
#                 if diff[c] > 0:
#                     changes.append(f"+{diff[c]} {CLASSES[c]}")
#                 elif diff[c] < 0:
#                     changes.append(f"{diff[c]} {CLASSES[c]}")
#             change_text = ", ".join(changes) if changes else "no change"

#             # 이미지 오버레이
#             img_path = os.path.join(img_dir, f"testset_event_{eid:05d}_2.jpg")
#             img = cv2.imread(img_path)
#             if img is None:
#                 prev = curr
#                 continue

#             org_y, dy, font = 30, 25, cv2.FONT_HERSHEY_SIMPLEX
#             cv2.putText(img, f"Event:{eid} W:{w} K:{k}", (10, org_y), font, 0.8, (0,255,0), 2)
#             cv2.putText(img, f"Change:{change_text}", (10, org_y+dy), font, 0.5, (0,255,255), 2)

#             line = 2
#             for c in range(num_classes):
#                 if curr[c] > 0:
#                     txt = f"{CLASSES[c]}:{curr[c]}"
#                     cv2.putText(img, txt, (10, org_y+(line+1)*dy), font, 0.6, (255,255,255), 1)
#                     line += 1

#             out_path = os.path.join(out_dir, f"event_{eid:05d}.jpg")
#             cv2.imwrite(out_path, img)
#             prev = curr

#         print(f"[DONE] win={w}, K={k} → {out_dir}")
