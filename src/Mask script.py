import os
import numpy as np
import matplotlib.pyplot as plt

from skimage.color import rgb2gray
from skimage import morphology
from skimage.filters import gaussian


# ---------------------------
# Load data
# ---------------------------
def load_image(path):
    return plt.imread(path)


def load_mask(path):
    gt = plt.imread(path)

    # If RGB → convert to grayscale by taking one channel
    if len(gt.shape) == 3:
        gt = gt[:, :, 0]

    # Normalize if needed (0–255 → 0–1)
    if gt.max() > 1:
        gt = gt / 255

    # Convert to binary
    gt = (gt > 0.5).astype(int)

    return gt


# ---------------------------
# Dice score
# ---------------------------
def calculate_dice(mask, gt):
    intersection = np.sum(mask * gt)
    dice = intersection * 2 / (np.sum(mask) + np.sum(gt))
    return dice


# ---------------------------
# Segmentation pipeline
# (following professor's notebook exactly)
# ---------------------------
def segment_image(im):
    # Remove alpha channel if present
    if im.shape[-1] == 4:
        im = im[:, :, :3]

    # Step 1: Convert to grayscale, scale to [0, 256]
    im256 = rgb2gray(im) * 256

    # Step 2: Gaussian blur (sigma=5)
    blurred_im = gaussian(im256, 5)

    # Step 3: Threshold — dark pixels = lesion
    mask = blurred_im < 120

    # Step 4: Opening — removes thin hairs and sharp noise at borders
    struct_el = morphology.disk(6)
    mask2 = morphology.binary_opening(mask, struct_el)

    # Step 5: Blur the mask to smooth edges
    blurred_mask = gaussian(mask2, 10)

    # Step 6: Re-binarize
    better_mask = blurred_mask > 0.5

    return better_mask.astype(np.uint8)


# ---------------------------
# Save mask
# ---------------------------
def save_mask(mask, path):
    plt.imsave(path, mask, cmap='gray')


# ---------------------------
# MAIN
# ---------------------------
if __name__ == "__main__":

    img_dir = "../data/imgs"
    mask_dir = "../data/masks"
    output_dir = "../data/output_masks"

    os.makedirs(output_dir, exist_ok=True)

    dice_scores = []

    for filename in os.listdir(img_dir):

        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        image_path = os.path.join(img_dir, filename)

        # Create mask filename
        name, _ = os.path.splitext(filename)
        gt_filename = name + "_mask.png"
        gt_path = os.path.join(mask_dir, gt_filename)

        # Output filename
        output_path = os.path.join(output_dir, gt_filename)

        # Load image
        im = load_image(image_path)

        # Segment
        pred_mask = segment_image(im)

        # Save result
        save_mask(pred_mask, output_path)

        # Evaluate if GT exists
        if os.path.exists(gt_path):
            gt = load_mask(gt_path)
            dice = calculate_dice(pred_mask, gt)
            dice_scores.append(dice)
            print(f"{filename} → Dice: {dice:.4f}")
        else:
            print(f"{filename} → mask saved (no GT found)")

    # Summary
    if dice_scores:
        print("\n--- Summary ---")
        print(f"Average Dice: {np.mean(dice_scores):.4f}")