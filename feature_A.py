import numpy as np
from PIL import Image
from scipy.ndimage import label
import os


#1. Load image 
def load_mask(path):
    img = Image.open(path).convert("L")
    mask = np.array(img)
    mask = (mask > 0).astype(np.uint8)  
    return mask


#2. Keep largest component 
def largest_component(mask):
    labeled, num = label(mask)

    if num == 0:
        return mask

    sizes = [(labeled == i).sum() for i in range(1, num + 1)]
    largest = np.argmax(sizes) + 1

    return (labeled == largest).astype(np.uint8)


#3. Crop to object
def crop_to_object(mask):
    ys, xs = np.where(mask == 1)

    if len(ys) == 0:
        return mask

    return mask[ys.min():ys.max()+1, xs.min():xs.max()+1]


#4. Split by centroid
def split_mask(mask):
    ys, xs = np.where(mask == 1)
    cy = int(np.mean(ys))
    cx = int(np.mean(xs))

    left  = mask[:, :cx]
    right = mask[:, cx:]

    top    = mask[:cy, :]
    bottom = mask[cy:, :]

    return left, right, top, bottom


#5. Align shapes (simple safe version)
def align(A, B):
    h = min(A.shape[0], B.shape[0])
    w = min(A.shape[1], B.shape[1])
    return A[:h, :w], B[:h, :w]


#6. Asymmetry score
def asymmetry(A, B):
    A = (A > 0).astype(np.uint8)
    B = (B > 0).astype(np.uint8)

    diff = np.abs(A.astype(int) - B.astype(int))

    return diff.sum() / (A.sum() + B.sum() + 1e-8)


#7. Main function 
def compute_asymmetry(path):
    mask = load_mask(path)
    mask = largest_component(mask)
    mask = crop_to_object(mask)

    left, right, top, bottom = split_mask(mask)

    # align
    left, right = align(left, right)
    top, bottom = align(top, bottom)

    # flip
    right = np.fliplr(right)
    bottom = np.flipud(bottom)

    # compute
    h = asymmetry(left, right)
    v = asymmetry(top, bottom)

    return {
        "horizontal": float(h),
        "vertical": float(v),
        "combined": float((h + v) / 2)
    }

folder = "data/masks"

for filename in os.listdir(folder):
    if filename.endswith(".png"):
        path = os.path.join(folder, filename)

        try:
            res = compute_asymmetry(path)

            print(f"{filename} -> H: {res['horizontal']:.3f}, "
                  f"V: {res['vertical']:.3f}, "
                  f"C: {res['combined']:.3f}")

        except Exception as e:
            print(f"Nope {filename}: {e}")
