import pandas as pd
import skimage as ski
import mahotas
import numpy as np
import os
from feature_B import compactness_calc

from color_features import extract_color_features

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The CSV just gotta have a img_id and diagnosis column
PATH = os.path.join(ROOT, "data", "metadata.csv")
df = pd.read_csv(PATH)

all_rows = []
# Filter away maskless images
mask_dir = os.path.join(ROOT, "data", "masks")

imgID = df["img_id"].to_list()

maskExists = []

for i in imgID:
    base_name = os.path.basename(i)
    name, ext = os.path.splitext(base_name)

    mask_path = os.path.join(mask_dir, f"{name}_mask{ext}")
    maskExists.append(os.path.exists(mask_path))

df = df[maskExists][["img_id", "diagnostic"]].reset_index(drop=True)

img_dir = os.path.join(ROOT, "data", "imgs")
mask_dir = os.path.join(ROOT, "data", "masks")

imgID = df["img_id"].to_list()
diagnosis = df["diagnostic"].to_list()
# ..........
#
#  Feature loop. Insert algos here
#
# ----------

for i in range(len(imgID)):
    img_path = os.path.join(img_dir, imgID[i])
    name, ext = os.path.splitext(os.path.basename(img_path))
    mask_path = os.path.join(mask_dir, f"{name}_mask{ext}")

    img = ski.io.imread(img_path)
    img = ski.transform.resize(img, (255, 255))

    mask = ski.io.imread(mask_path)
    mask = ski.transform.resize(mask, (255, 255), preserve_range=True)

    # mask to boolean
    if mask.ndim == 3:
        mask = ski.color.rgb2gray(mask) > 0.5
    else:
        mask = mask > 0.5

    #compactness
    compactness = compactness_calc(mask)
    compactnesses.append(compactness)
    
    feature_record = extract_color_features(img, mask)
    feature_record['compactness'] = compactness
    feature_record['img_id'] = imgID[i]
    feature_record['diagnostic'] = diagnosis[i]
    all_rows.append(feature_record)
    
df_features = pd.DataFrame(all_rows)

# 5. Save this new, complete CSV
output_path = os.path.join(ROOT, "data", "features.csv")
df_features.to_csv(output_path, index=False)
