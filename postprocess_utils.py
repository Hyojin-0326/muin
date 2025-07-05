# postprocess_utils.py
import numpy as np

def weighted_vote_union(counts, cam_weights, thr=1.0, min_hits=1):
    """
    1번 아이디어:
      - 존재 여부만 가중 합산해 thr 넘으면 “있음”으로 인정
      - 실제 개수는 counts.max(axis=1) 사용
    """
    present  = (counts > 0).astype(float)          # (C, K)
    vote_sum = (present * cam_weights).sum(axis=1) # (C,)
    valid    = (present.sum(axis=1) >= min_hits)   # (C,)
    result   = counts.max(axis=1)                 # (C,)
    return result * (vote_sum >= thr) * valid      # (C,)

def weighted_union_simple(counts, cam_weights, min_hits=1):
    """
    2번 아이디어:
      - 존재 여부만 가중 합산 → true/false 마스크
      - 그 마스크로 counts 합집합(union) 취함
    """
    present  = (counts > 0).astype(float)
    vote_sum = (present * cam_weights).sum(axis=1)
    valid    = (present.sum(axis=1) >= min_hits)
    mask     = vote_sum > 0
    # union = counts.max  OR  counts.sum? “합집합”이라면 max
    return counts.max(axis=1) * mask * valid

def pairwise_presence_union(counts, pairs, min_hits=1):
    """
    3번 아이디어:
      - pairs: [(0,4), (1,3), ...]
      - 쌍(pair)마다 둘 다 검출된 클래스만 존재표 true
      - 최종 개수는 counts.max(axis=1)
    """
    C, K = counts.shape
    present = counts > 0
    # 각 클래스가 “어떤 쌍”에서도 한번이라도 둘 다 검출됐으면 true
    pair_mask = np.zeros(C, dtype=bool)
    for (i,j) in pairs:
        both = present[:,i] & present[:,j]
        pair_mask |= both
    return counts.max(axis=1) * pair_mask * (present.sum(axis=1) >= min_hits)

def quadrant_presence_union(boxes_by_cam, image_size, cam_map, min_hits=1, num_classes=60):
    """
    4번 아이디어:
      - boxes_by_cam: dict[cam_idx] → list of (cls, center_x, center_y)
      - image_size: (W, H)
      - cam_map: { quadrant_index: cam_idx }
      - min_hits: 한 쿼드런트에서 최소 검출 횟수
      - num_classes: 전체 클래스 수
    """
    W, H = image_size
    from collections import defaultdict
    cls_quads = defaultdict(set)
    for quad, cam in cam_map.items():
        for cls, x, y in boxes_by_cam.get(cam, []):
            qx = 0 if x < W/2 else 1
            qy = 0 if y < H/2 else 1
            if (qy*2 + qx) == quad:
                cls_quads[cls].add(cam)

    # 전체 클래스 수로 mask 생성
    mask = np.zeros(num_classes, dtype=bool)
    for cls, cams in cls_quads.items():
        if len(cams) >= min_hits and cls < num_classes:
            mask[cls] = True

    return mask

def quadrant_presence_max(boxes_by_cam, counts, image_size, cam_map, min_hits=1,num_classes=60):
    """
    5번 아이디어:
      - 4번과 같지만 최종 개수는 counts.max(axis=1)
    """
    mask = quadrant_presence_union(boxes_by_cam, image_size, cam_map, min_hits, num_classes)
    return counts.max(axis=1) * mask
