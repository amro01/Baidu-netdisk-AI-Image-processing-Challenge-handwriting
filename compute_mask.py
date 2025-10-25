import cv2
import os
import glob
import numpy as np
from tqdm import tqdm

def generate_masks_for_dir(image_dir, gt_dir, save_path, suffix, gt_suffix='.png'):
    """
    Generates masks for a given directory of images.

    Args:
        image_dir (str): Path to the directory containing input images.
        gt_dir (str): Path to the directory containing ground truth images.
        save_path (str): Path to save the generated masks.
        suffix (str): The file suffix of the input images (e.g., '.jpg' or '_augmented.png').
    """
    print(f"Processing directory: {image_dir}")
    image_files = glob.glob(os.path.join(image_dir, f'*{suffix}'))
    
    if not image_files:
        print(f"Warning: No images found in {image_dir} with suffix '{suffix}'")
        return

    for im_f in tqdm(image_files, desc=f"Generating masks for {os.path.basename(image_dir)}"):
        base_name = os.path.basename(im_f)
        
        # Construct the corresponding ground truth file path
        if suffix == '_augmented.png':
            gt_base_name = base_name.replace('_augmented.png', '.png')
        else:
            gt_base_name = base_name.replace(suffix, gt_suffix)
            
        gt_f = os.path.join(gt_dir, gt_base_name)

        if not os.path.exists(gt_f):
            print(f"Warning: Ground truth file not found for {im_f}, skipping.")
            continue

        gt = cv2.imread(gt_f)
        im = cv2.imread(im_f)
        
        if gt is None or im is None:
            print(f"Warning: Could not read image or GT for {base_name}, skipping.")
            continue

        kernel = np.ones((3,3), np.uint8) 
        threshold = 25
        diff_image = np.abs(im.astype(np.float32) - gt.astype(np.float32))
        mean_image = np.mean(diff_image, axis=-1)
        mask = np.greater(mean_image, threshold).astype(np.uint8)
        mask = (1 - mask) * 255
        mask = cv2.erode(np.uint8(mask), kernel, iterations=1)
        
        # Save the mask with the same name as the input image (but as a .png)
        save_mask_path = os.path.join(save_path, base_name.split('.')[0] + '.png')
        if suffix == '_augmented.png':
            save_mask_path = os.path.join(save_path, base_name)

        cv2.imwrite(save_mask_path, np.uint8(mask))

if __name__ == '__main__':
    # Base path for the finetuning dataset
    base_path = './finetune_dataset'
    save_path = os.path.join(base_path, 'mask/')
    os.makedirs(save_path, exist_ok=True)

    # --- Generate masks for TRAINING set ---
    print("--- Generating masks for TRAINING set ---")
    # For original images (.jpg)
    train_images_dir = os.path.join(base_path, 'images')
    train_gts_dir = os.path.join(base_path, 'gts')
    generate_masks_for_dir(train_images_dir, train_gts_dir, save_path, suffix='.jpg', gt_suffix='.png')
    
    # For augmented images (_augmented.png)
    train_aug_dir = os.path.join(base_path, 'augmented_images')
    generate_masks_for_dir(train_aug_dir, train_gts_dir, save_path, suffix='_augmented.png', gt_suffix='.png')
    
    print("\n--- Generating masks for VALIDATION set ---")
    # --- Generate masks for VALIDATION set ---
    val_images_dir = os.path.join(base_path, 'val_images')
    val_gts_dir = os.path.join(base_path, 'val_gts')
    # For original validation images (.jpg)
    generate_masks_for_dir(val_images_dir, val_gts_dir, save_path, suffix='.jpg', gt_suffix='.png')
    
    # For augmented validation images (_augmented.png)
    generate_masks_for_dir(val_images_dir, val_gts_dir, save_path, suffix='_augmented.png', gt_suffix='.png')

    print("\nMask generation complete.")
    print(f"Masks are saved in: {save_path}")
