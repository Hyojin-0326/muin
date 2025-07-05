import numpy as np
import os
import utils_yolo_nas
import csv
import shutil



# detection 함수랑 CLASSES는 따로 임포트되어 있다고 가정


img_folder = "/home/aistore51/Datasets/5.eval_testset" 
output_dir = "/home/aistore51/git/results"
os.makedirs(output_dir, exist_ok=True)

num_classes = 60
num_cams = 5
start_event = 1
end_event = 333
#price_dict  

# 카메라별 원본 폴더
cam_folders = [os.path.join(img_folder, f"cam{i}") for i in range(num_cams)]

for event_id in range(start_event, end_event + 1):
    print(f"[INFO] Processing event {event_id}")

    # 1) 임시 폴더에 각 cam 이미지 복사
    temp_cam_folders = []
    for cam_idx, cam_folder in enumerate(cam_folders):
        src = os.path.join(cam_folder, f"testset_event_{event_id:05d}_{cam_idx}.jpg")
        if not os.path.isfile(src):
            print(f"[WARNING] 이미지 없음: {src}")
            continue

        tmp_dir = os.path.join("/tmp", f"event_{event_id:05d}_cam_{cam_idx}")
        os.makedirs(tmp_dir, exist_ok=True)
        shutil.copy2(src, tmp_dir)
        temp_cam_folders.append(tmp_dir)

    if not temp_cam_folders:
        print(f"[INFO] Event {event_id}에 사용할 이미지가 하나도 없음, 스킵함")
        continue

    # 2) detection 호출 (positional arg or cam_inputs=)
    counts, boxes_by_cam = utils_yolo_nas.detection(temp_cam_folders)
    # 또는: utils_yolo_nas.detection(cam_inputs=temp_cam_folders)

    # 3) 이벤트별 .txt 저장
    out_path = os.path.join(output_dir, f"event_{event_id:05d}.txt")
    with open(out_path, "w", newline="") as fw:
        writer = csv.writer(fw, delimiter='\t')

        # 3-1) counts
        header = ["cls_id"] + [f"cam{i}" for i in range(len(temp_cam_folders))]
        writer.writerow(header)
        for cls_id in range(counts.shape[0]):
            if counts[cls_id].sum() == 0:
                continue
            row = [f"{cls_id:02d}"] + counts[cls_id].tolist()
            writer.writerow(row)

        # 3-2) boxes
        writer.writerow([])
        writer.writerow(["cam_idx", "cls_id", "cx", "cy"])
        for cam_idx, boxlist in boxes_by_cam.items():
            for cls, cx, cy in boxlist:
                writer.writerow([cam_idx, f"{cls:02d}", f"{cx:.1f}", f"{cy:.1f}"])

    # 4) 임시 폴더 정리
    for d in temp_cam_folders:
        shutil.rmtree(d)