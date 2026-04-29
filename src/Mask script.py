import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray, rgb2lab
from skimage import morphology
from skimage.filters import gaussian, sobel
from skimage.measure import label
from skimage.segmentation import active_contour
import warnings


# ---------------------------
# Data loading
# ---------------------------

def load_image(path: str) -> np.ndarray:
    """Load an RGB image as a float array in [0, 1]."""
    im = plt.imread(path)
    if im.dtype == np.uint8:
        im = im / 255.0
    im = im.astype(np.float64)
    if im.ndim == 3 and im.shape[2] == 4:
        im = im[..., :3]
    return im


def load_mask(path: str) -> np.ndarray:
    """
    Load a binary mask and normalise to {0, 1}.
    Handles both 0-1 float and 0-255 uint8 masks.
    """
    gt = plt.imread(path)
    if gt.max() > 1:
        gt = gt / 255.0
    if gt.ndim == 3:
        gt = gt[..., 0]
    return (gt > 0.5).astype(np.uint8)


# ---------------------------
# Evaluation
# ---------------------------

def calculate_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Sorensen-Dice coefficient between two binary masks."""
    intersection = np.sum(pred * gt)
    return (2.0 * intersection) / (np.sum(pred) + np.sum(gt) + 1e-8)


# ---------------------------
# Edge map
# ---------------------------

def build_edge_map(im: np.ndarray) -> np.ndarray:
    """
    Build an edge-strength map from the image.

    We use the LAB colour space rather than grayscale so that colour
    transitions (e.g. brown lesion on pale skin) are captured even when
    the lightness contrast is low.  The edge map is the mean of Sobel
    responses across all three LAB channels, each normalised to [0, 1].

    A strong edge (value near 1) means the image has a sharp boundary
    there — exactly where we want the refined mask edge to sit.

    Returns
    -------
    np.ndarray  shape (H, W), float [0, 1], higher = stronger edge.
    """
    lab      = rgb2lab(im)
    channels = [
        lab[..., 0] / 100.0,           # L
        (lab[..., 1] + 128) / 255.0,   # A
        (lab[..., 2] + 128) / 255.0,   # B
    ]
    edge_maps = []
    for ch in channels:
        blurred  = gaussian(ch, sigma=2)
        edges    = sobel(blurred)
        # Normalise each channel's edge map to [0, 1].
        e_min, e_max = edges.min(), edges.max()
        if e_max > e_min:
            edges = (edges - e_min) / (e_max - e_min)
        edge_maps.append(edges)

    return np.mean(edge_maps, axis=0)


# ---------------------------
# Mask refinement
# ---------------------------

def resample_contour(contour: np.ndarray, n_points: int) -> np.ndarray:
    """
    Resample an (N, 2) contour to exactly n_points via linear interpolation.
    """
    from scipy.interpolate import interp1d
    t_orig = np.linspace(0, 1, len(contour))
    t_new  = np.linspace(0, 1, n_points)
    return interp1d(t_orig, contour, axis=0)(t_new)


def extract_blob_contours(mask: np.ndarray, n_points: int = 200) -> list:
    """
    Extract one contour per connected component in the mask.

    Previously only the largest blob was extracted, so any smaller blobs
    were silently dropped before refinement. Now every component gets its
    own contour and its own snake pass, so all blobs are preserved and
    refined independently.

    Parameters
    ----------
    mask     : binary mask, shape (H, W).
    n_points : number of points to resample each contour to.

    Returns
    -------
    list of np.ndarray, each shape (n_points, 2) in (row, col) coordinates.
      One entry per connected component found in the mask.
    """
    from skimage.measure import find_contours

    labeled    = label(mask)
    n_blobs    = labeled.max()
    H, W       = mask.shape
    contours_out = []

    for blob_id in range(1, n_blobs + 1):
        blob     = (labeled == blob_id).astype(np.uint8)
        contours = find_contours(blob, 0.5)
        if not contours:
            continue
        # Take the outermost (longest) contour for this blob.
        contour = max(contours, key=len)
        if len(contour) < 4:
            # Too small to resample meaningfully — keep as-is via rasterisation.
            contours_out.append(contour)
            continue
        contours_out.append(resample_contour(contour, n_points))

    if not contours_out:
        # Empty mask fallback: single small circle at centre.
        t = np.linspace(0, 2*np.pi, n_points)
        r = min(H, W) * 0.1
        contours_out.append(
            np.column_stack([H/2 + r*np.sin(t), W/2 + r*np.cos(t)])
        )

    return contours_out


def snap_contour_to_edges(
    snake_init: np.ndarray,
    edge_map: np.ndarray,
    alpha: float = 0.01,
    beta: float  = 1.0,
    gamma: float = 0.01,
    max_iter: int = 500,
) -> np.ndarray:
    """
    Run skimage's active_contour (snake) algorithm to snap the initial
    contour to nearby image edges.

    The snake is attracted to bright regions of `edge_map` (strong image
    edges) and repelled by the smoothness constraints (alpha, beta) that
    prevent it collapsing or developing kinks.

    Parameters
    ----------
    snake_init : (N, 2) initial contour in (row, col) coordinates.
    edge_map   : (H, W) float edge-strength image, higher = stronger edge.
    alpha      : contour length weight (elasticity). Higher = shorter snake.
    beta       : contour smoothness weight (rigidity). Higher = smoother.
    gamma      : step size. Lower = more careful, slower convergence.
    max_iter   : maximum number of iterations.

    Returns
    -------
    np.ndarray  shape (N, 2), refined contour in (row, col) coordinates.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        snake = active_contour(
            edge_map,
            snake_init,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            max_num_iter=max_iter,
            boundary_condition="periodic",
        )
    return snake


def contour_to_mask(contour: np.ndarray, shape: tuple) -> np.ndarray:
    """
    Convert an (N, 2) (row, col) contour back into a filled binary mask.

    Uses matplotlib's path fill to rasterise the polygon.

    Returns
    -------
    np.ndarray  shape (H, W), dtype uint8, values in {0, 1}.
    """
    from matplotlib.path import Path

    H, W   = shape
    path   = Path(contour[:, ::-1])   # Path expects (x=col, y=row)
    cols, rows = np.meshgrid(np.arange(W), np.arange(H))
    points = np.column_stack([cols.ravel(), rows.ravel()])
    mask   = path.contains_points(points).reshape(H, W)
    return mask.astype(np.uint8)


def morphological_cleanup(mask: np.ndarray, disk_radius: int = 3) -> np.ndarray:
    """
    Light morphological closing to fill any small holes left after
    contour rasterisation, followed by opening to smooth rough edges.

    We use a smaller disk (radius 3) than before because the contour
    is already well-positioned — we only want minor tidying.
    """
    struct_el = morphology.disk(disk_radius)
    mask = morphology.binary_closing(mask, struct_el)
    mask = morphology.binary_opening(mask, struct_el)
    return mask.astype(np.uint8)


# ---------------------------
# Quality check
# ---------------------------

def boundary_edge_score(mask: np.ndarray, edge_map: np.ndarray) -> float:
    """
    Measure how well the mask boundary already aligns with image edges.

    We dilate and erode the mask to produce a thin band of pixels along its
    boundary, then sample the edge map values at those pixels. The mean edge
    strength there is the alignment score.

    A score near 1.0 means the boundary sits exactly on sharp image edges
    (mask is already good). A low score means the boundary crosses through
    uniform regions (mask needs adjustment).

    Parameters
    ----------
    mask     : binary mask, shape (H, W).
    edge_map : edge-strength image, shape (H, W), float [0, 1].

    Returns
    -------
    float  Mean edge strength along the mask boundary, in [0, 1].
    """
    ring = morphology.disk(2)
    dilated  = morphology.binary_dilation(mask, ring)
    eroded   = morphology.binary_erosion(mask, ring)
    boundary = dilated & ~eroded          # thin band around the mask edge
    if boundary.sum() == 0:
        return 0.0
    return float(edge_map[boundary].mean())


def mask_needs_refinement(
    mask: np.ndarray,
    edge_map: np.ndarray,
    threshold: float = 0.2,
) -> bool:
    """
    Return True if the mask boundary is poorly aligned with image edges
    and should be refined, False if it is already good enough to skip.

    The threshold of 0.2 means: if on average the boundary sits on pixels
    with at least 20% of the maximum edge strength in the image, we
    consider it acceptable. You can raise this to be more aggressive (refine
    more images) or lower it to be more conservative (skip more images).

    Parameters
    ----------
    mask      : binary mask, shape (H, W).
    edge_map  : edge-strength image, shape (H, W), float [0, 1].
    threshold : minimum acceptable boundary edge score (default 0.2).

    Returns
    -------
    bool
    """
    score = boundary_edge_score(mask, edge_map)
    return score < threshold


# ---------------------------
# Refinement pipeline
# ---------------------------

def refine_mask(im: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Refine an existing binary mask by snapping each blob boundary to image edges.

    Pipeline
    --------
    1. Build a colour edge map from the image (LAB Sobel).
    2. Extract one contour per connected component in the mask.
    3. Run active contour (snake) independently on each blob contour.
    4. Rasterise each refined contour back to a binary mask.
    5. OR all per-blob masks together so no blob is lost.
    6. Light morphological clean-up.

    Processing blobs independently means masks with multiple lesions are
    fully preserved — previously only the largest blob was refined and
    all others were silently dropped.

    Parameters
    ----------
    im   : RGB float [0,1], shape (H, W, 3).
    mask : existing binary mask, shape (H, W), values {0, 1}.

    Returns
    -------
    np.ndarray  Refined binary mask, dtype uint8, shape (H, W).
    """
    edge_map = build_edge_map(im)

    # Build the edge map once and pass it in so the quality check and the
    # snake refinement share the same computation.
    blobs    = extract_blob_contours(mask)
    combined = np.zeros(im.shape[:2], dtype=np.uint8)

    for snake_init in blobs:
        snake     = snap_contour_to_edges(snake_init, edge_map)
        blob_mask = contour_to_mask(snake, im.shape[:2])
        combined  = np.maximum(combined, blob_mask)   # OR across blobs

    combined = morphological_cleanup(combined)
    return combined, edge_map


# ---------------------------
# Output
# ---------------------------

def save_mask(mask: np.ndarray, path: str) -> None:
    """Save a binary {0,1} mask as a grayscale PNG."""
    plt.imsave(path, mask, cmap="gray")


# ---------------------------
# Main
# ---------------------------

def main() -> None:

    img_dir    = "../data/imgs"
    mask_dir   = "../data/masks"        # input masks to refine
    output_dir = "../data/output_masks" # refined masks + images saved here

    os.makedirs(output_dir, exist_ok=True)

    dice_scores: list[float] = []
    n_deleted  = 0
    n_skipped  = 0
    n_refined  = 0

    for filename in sorted(os.listdir(img_dir)):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        image_path    = os.path.join(img_dir, filename)
        name, ext     = os.path.splitext(filename)
        mask_filename = name + "_mask.png"
        mask_path     = os.path.join(mask_dir, mask_filename)

        # No mask found — delete the image from the source directory.
        if not os.path.exists(mask_path):
            os.remove(image_path)
            n_deleted += 1
            print(f"{filename} -> no mask found, image deleted")
            continue

        out_mask_path = os.path.join(output_dir, mask_filename)

        im         = load_image(image_path)
        input_mask = load_mask(mask_path)

        # Quick quality check: compute the edge map (cheap) and measure how
        # well the existing mask boundary already aligns with image edges.
        # If the score is above the threshold the mask is good enough — copy
        # both the image and mask to output without running the snake
        # (which is the expensive part).
        edge_map = build_edge_map(im)
        if not mask_needs_refinement(input_mask, edge_map):
            save_mask(input_mask, out_mask_path)
            n_skipped += 1
            print(f"{filename} -> mask already good, copied to output")
            continue

        # Mask needs work — refine and copy both files to output.
        refined_mask, _ = refine_mask(im, input_mask)
        save_mask(refined_mask, out_mask_path)

        dice = calculate_dice(refined_mask, input_mask)
        dice_scores.append(dice)
        n_refined += 1
        print(f"{filename} -> refined, Dice vs input: {dice:.4f}")

    print("\n--- Summary ---")
    print(f"Deleted (no mask)  : {n_deleted}")
    print(f"Copied as-is       : {n_skipped}")
    print(f"Refined            : {n_refined}")
    if dice_scores:
        print(f"Avg Dice vs input  : {np.mean(dice_scores):.4f}")
        print(f"Std                : {np.std(dice_scores):.4f}")
        print(f"Min / Max          : {np.min(dice_scores):.4f} / {np.max(dice_scores):.4f}")


if __name__ == "__main__":
    main()