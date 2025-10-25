import os
import glob
import random
import shutil
from tqdm import tqdm

def clean_and_prepare_dataset(base_dir, val_split_count=80):
    """
    Cleans, verifies, and splits the dataset to ensure consistency.

    0. Restores any gts files from val_gts back to the main gts directory.
    1. Deletes existing val_images and val_gts directories.
    2. Verifies that every image in 'images' has a corresponding gt before the split.
    3. Creates a new validation set by moving a random sample of original images and their gts.
    4. Verifies 'augmented_images' against the remaining training gts and cleans it.
    """
    images_dir = os.path.join(base_dir, 'images')
    augmented_dir = os.path.join(base_dir, 'augmented_images')
    gts_dir = os.path.join(base_dir, 'gts')
    val_images_dir = os.path.join(base_dir, 'val_images')
    val_gts_dir = os.path.join(base_dir, 'val_gts')

    def verify_and_clean(image_folder, gt_folder, suffix, gt_suffix='.png'):
        """Helper function to verify images against gts and remove orphans."""
        print(f"Verifying files in: {image_folder}")
        image_files = glob.glob(os.path.join(image_folder, f'*{suffix}'))
        files_to_remove = []

        for img_path in tqdm(image_files, desc=f"Checking {os.path.basename(image_folder)}"):
            base_name = os.path.basename(img_path)
            
            if suffix == '_augmented.png':
                gt_base_name = base_name.replace('_augmented.png', '.png')
            else:
                gt_base_name = base_name.replace(suffix, gt_suffix)
            
            gt_path = os.path.join(gt_folder, gt_base_name)
            
            if not os.path.exists(gt_path):
                files_to_remove.append(img_path)
        
        if files_to_remove:
            print(f"Found {len(files_to_remove)} images in {os.path.basename(image_folder)} without corresponding ground truth. Removing them...")
            for f in files_to_remove:
                os.remove(f)
        else:
            print(f"All images in {os.path.basename(image_folder)} have corresponding ground truth files.")

    # --- 0. Restore validation gts to the main pool ---
    print("--- Step 0: Restoring gts from validation set (if any) ---")
    if os.path.exists(val_gts_dir):
        val_gts_files = glob.glob(os.path.join(val_gts_dir, '*.png'))
        if val_gts_files:
            for f in tqdm(val_gts_files, desc="Restoring gts"):
                shutil.move(f, gts_dir)
            print(f"Restored {len(val_gts_files)} gts files.")
    else:
        print("Validation gts directory does not exist, skipping restore.")

    # --- 1. Clean up old validation directories ---
    print("\n--- Step 1: Cleaning up old validation directories ---")
    for d in [val_images_dir, val_gts_dir]:
        if os.path.exists(d):
            shutil.rmtree(d)
    os.makedirs(val_images_dir)
    os.makedirs(val_gts_dir)
    print("Created new, empty validation directories.")

    # --- 2. Pre-verification of original images ---
    print("\n--- Step 2: Pre-verifying original 'images' set ---")
    verify_and_clean(images_dir, gts_dir, '.jpg')

    # --- 3. Create the validation split ---
    print("\n--- Step 3: Creating a new validation split ---")
    all_original_images = glob.glob(os.path.join(images_dir, '*.jpg'))
    if len(all_original_images) < val_split_count:
        val_split_count = len(all_original_images)
        print(f"Warning: Not enough images. Validation set size adjusted to {val_split_count}.")

    validation_set = random.sample(all_original_images, val_split_count)
    print(f"Randomly selected {len(validation_set)} images for the new validation set.")

    for img_path in tqdm(validation_set, desc="Moving validation files"):
        base_name = os.path.basename(img_path)
        gt_base_name = base_name.replace('.jpg', '.png')
        gt_path = os.path.join(gts_dir, gt_base_name)

        shutil.move(img_path, val_images_dir)
        if os.path.exists(gt_path):
            shutil.move(gt_path, val_gts_dir)
        else:
            print(f"CRITICAL WARNING: Could not find gt for moved image: {base_name}")

    # --- 4. Post-split verification of augmented images ---
    print("\n--- Step 4: Verifying 'augmented_images' against remaining training gts ---")
    verify_and_clean(augmented_dir, gts_dir, '_augmented.png')

    print("\n--- Dataset Preparation Summary ---")
    print(f"Total original training images left: {len(glob.glob(os.path.join(images_dir, '*.jpg')))}")
    print(f"Total augmented training images: {len(glob.glob(os.path.join(augmented_dir, '*.png')))}")
    print(f"Total ground truth files for training: {len(glob.glob(os.path.join(gts_dir, '*.png')))}")
    print(f"Total validation images: {len(glob.glob(os.path.join(val_images_dir, '*.jpg')))}")
    print(f"Total validation gts: {len(glob.glob(os.path.join(val_gts_dir, '*.png')))}")
    print("\nDataset preparation complete.")


if __name__ == '__main__':
    dataset_base_path = './finetune_dataset'
    clean_and_prepare_dataset(dataset_base_path)