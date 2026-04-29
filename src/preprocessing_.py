# Project: PDS 2026 - Skin Lesion Classification - Group Quokka
# Hair and marker removal from skin lesion images.

import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage import morphology, restoration, util


INPUT_FOLDER  = "/Users/jakubstopiak/Desktop/IMGT"
MASK_FOLDER   = "/Users/jakubstopiak/Desktop/bordel/IMGT_masks"
OUTPUT_FOLDER = "/Users/jakubstopiak/Desktop/IMGT_clean"


def remove_hair(img, lesion_mask):
    """Remove hair: aggressive on the skin, gentle on the lesion."""
    gray = rgb2gray(img)

    blackhat = morphology.black_tophat(gray, morphology.disk(5))
    tophat   = morphology.white_tophat(gray, morphology.disk(5))

    # Skin uses 0.04, lesion uses stricter 0.10 (only strong hair on the lesion)
    threshold = np.where(lesion_mask > 0, 0.10, 0.04)
    mask = (blackhat > threshold) | (tophat > threshold)
    mask = morphology.binary_dilation(mask, morphology.disk(1))

    return inpaint(img, mask)


def remove_marker(img, lesion_mask):
    """Remove thick dark marker ink lines, but only outside the lesion."""
    gray = rgb2gray(img)

    blackhat = morphology.black_tophat(gray, morphology.disk(25))
    mask = blackhat > 0.04
    mask = morphology.binary_dilation(mask, morphology.disk(8))

    mask[lesion_mask > 0] = False

    return inpaint(img, mask)


def inpaint(img, mask):
    """Fill the masked pixels by interpolating from surrounding pixels."""
    if not mask.any():
        return util.img_as_float(img)
    return restoration.inpaint_biharmonic(util.img_as_float(img), mask, channel_axis=-1)


def clean_image(img, lesion_mask):
    """Run the full pipeline: hair removal, then marker removal."""
    img = remove_hair(img, lesion_mask)
    img = remove_marker(img, lesion_mask)
    return img


def process_folder(input_folder, mask_folder, output_folder):
    """Clean every image in the folder and save with a 'clean_' prefix."""
    os.makedirs(output_folder, exist_ok=True)
    files = [f for f in os.listdir(input_folder) if f.endswith(".png")]

    for i, filename in enumerate(files):
        print(f"[{i + 1}/{len(files)}] {filename}")

        img = plt.imread(os.path.join(input_folder, filename))
        if img.shape[-1] == 4:
            img = img[:, :, :3]
        mask_name = filename.replace(".png", "_mask.png")
        lesion_mask = plt.imread(os.path.join(mask_folder, mask_name)) > 0.5

        cleaned = clean_image(img, lesion_mask)
        plt.imsave(os.path.join(output_folder, "clean_" + filename), cleaned)


if __name__ == "__main__":
    process_folder(INPUT_FOLDER, MASK_FOLDER, OUTPUT_FOLDER)
