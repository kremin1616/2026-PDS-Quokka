from skimage import color
import numpy as np

def extract_color_features(img_rgb, mask):
    """
    Extract colour features from a skin lesion image.
    
    Parameters:
        img_rgb : np.ndarray, shape (H, W, 3), dtype float, values 0.0-1.0
        mask    : np.ndarray, shape (H, W), dtype bool, True = lesion pixel
    
    Returns:
        dict of feature_name -> float value
    """
    lesion_pixels = img_rgb[mask]      
    surrounding   = img_rgb[~mask]     

    FEATURE_NAMES = [
        'rgb_R_mean', 'rgb_G_mean', 'rgb_B_mean',
        'rgb_R_std',  'rgb_G_std',  'rgb_B_std',
        'hsv_H_mean', 'hsv_S_mean', 'hsv_V_mean',
        'lab_L_mean', 'lab_A_mean', 'lab_B_mean',
        'color_nonuniformity',
        'rel_contrast'
    ]

    if len(lesion_pixels) == 0:
       
        return {k: np.nan for k in FEATURE_NAMES}

    features = {}


    for i, ch in enumerate(['R', 'G', 'B']):
        features[f'rgb_{ch}_mean'] = lesion_pixels[:, i].mean()
        features[f'rgb_{ch}_std']  = lesion_pixels[:, i].std()
      
    img_hsv    = color.rgb2hsv(img_rgb)
    lesion_hsv = img_hsv[mask]
    features['hsv_H_mean'] = lesion_hsv[:, 0].mean()  
    features['hsv_S_mean'] = lesion_hsv[:, 1].mean()  
    features['hsv_V_mean'] = lesion_hsv[:, 2].mean() 

  
    lesion_lab = img_lab[mask]
    features['lab_L_mean'] = lesion_lab[:, 0].mean()  
    features['lab_A_mean'] = lesion_lab[:, 1].mean()  
    features['lab_B_mean'] = lesion_lab[:, 2].mean()  
   
    features['color_nonuniformity'] = lesion_pixels.std(axis=0).mean()
   
   #this is an extra feature that idk if we should use because it wfgwai
    if len(surrounding) > 0:
        features['rel_contrast'] = lesion_pixels.mean() - surrounding.mean()
    else:
        features['rel_contrast'] = np.nan

    return features
