import os
import numpy as np
import utils 
import csv

# 설정
img_folder = "/home/aistore51/Datasets/4.testset_sample" 
output_dir = "/home/aistore51/git/output"
os.makedirs(output_dir, exist_ok=True)

num_classes = 60
num_cams = 5
start_event = 10001
end_event = 10175

# 카메라별 폴더 구성
cam_folders = [
    os.path.join(img_folder, f"cam{i}") for i in range(num_cams)
]

# 메인 루프
for event_id in range(start_event, end_event + 1):
    print(f"[INFO] Processing event {event_id}")

    # 해당 이벤트의 이미지를 카메라별로 모아 임시 폴더 구성
    temp_cam_folders = []
    for cam_idx in range(num_cams):
        temp_dir = f"/tmp/event_{event_id}_cam_{cam_idx}"
        os.makedirs(temp_dir, exist_ok=True)

        src_img = os.path.join(cam_folders[cam_idx], f"testset_event_{event_id:05d}_{cam_idx}.jpg")
        dst_img = os.path.join(temp_dir, f"testset_event_{event_id:05d}_{cam_idx}.jpg")
        os.system(f"cp {src_img} {dst_img}")
        temp_cam_folders.append(temp_dir)

    # output.txt 경로 설정
    output_txt_path = os.path.join(output_dir, f"event_{event_id:05d}.txt")

    # detection_voting 실행
    utils.detection_voting(
        cam_folders=temp_cam_folders,
        output_txt_path=output_txt_path
    )

    # 임시 폴더 제거
    for folder in temp_cam_folders:
        os.system(f"rm -r {folder}")
    
