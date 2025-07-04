import utils
import numpy as np

import os



# %%
cam_folders = [
    f"/home/aistore51/Datasets/4.testset_sample/cam{i}/" for i in range(5)
]
num_classes = 60
num_cams = len(cam_folders)

# %%


final_counts = utils.detection_voting(
    cam_folders=cam_folders
)

print("✅ detection voting 완료됨!")