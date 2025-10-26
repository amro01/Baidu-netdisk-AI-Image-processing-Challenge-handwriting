import os
import glob
import random
import shutil
from tqdm import tqdm

def clean_and_prepare_dataset(base_dir, val_split_count=80):
    """
    Cleans, verifies, and splits dataset to ensure consistency.

    0. Restores any gts files from val_gts back to main gts directory.
    1. Deletes existing val_images and val_gts directories.
    2. Verifies that every image in 'images' has a corresponding gt before the split.
    3. Creates a new validation set by moving a random sample of original images and their gts.
    4. Verifies 'augmented_images' against remaining training gts and cleans it.
    """
    images_dir = os.path.join(base_dir, 'images')
    augmented_dir = os.path.join(base_dir, 'augmented_images')
    gts_dir = os.path.join(base_dir, 'gts')
    val_images_dir = os.path.join(base_dir, 'val_images')
    val_gts_dir = os.path.join(base_dir, 'val_gts')
    val_augmented_dir = os.path.join(base_dir, 'val_augmented_images')

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

    # --- 0. Restore validation set to the main pool ---
    print("--- Step 0: Restoring validation set (if any) ---")
    # Restore validation images
    if os.path.exists(val_images_dir):
        val_img_files = glob.glob(os.path.join(val_images_dir, '*')) # Handles both .jpg and .png
        if val_img_files:
            for f in tqdm(val_img_files, desc="Restoring images"):
                # Move .jpg back to images/, .png back to augmented_images/
                if f.endswith('.jpg'):
                    shutil.move(f, images_dir)
                elif f.endswith('_augmented.png'):
                    shutil.move(f, augmented_dir)
            print(f"Restored {len(val_img_files)} image files.")
    else:
        print("Validation images directory does not exist, skipping restore.")

    # Restore validation gts
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
    
    # Group images by their base name (e.g., dehw_train_00520)
    image_groups = {}
    for img_path in all_original_images:
        base_name = os.path.basename(img_path).replace('.jpg', '')
        image_groups[base_name] = {'original': img_path}
        augmented_path = os.path.join(augmented_dir, f'{base_name}_augmented.png')
        if os.path.exists(augmented_path):
            image_groups[base_name]['augmented'] = augmented_path

    if len(image_groups) < val_split_count:
        val_split_count = len(image_groups)
        print(f"Warning: Not enough image groups. Validation set size adjusted to {val_split_count}.")

    # Randomly select groups for validation
    group_names = list(image_groups.keys())
    validation_groups = random.sample(group_names, val_split_count)
    print(f"Randomly selected {len(validation_groups)} image groups for the new validation set.")

    for group_name in tqdm(validation_groups, desc="Moving validation files"):
        group = image_groups[group_name]
        gt_base_name = f'{group_name}.png'
        gt_path = os.path.join(gts_dir, gt_base_name)

        # Decide whether to use the augmented or original image for validation (e.g., 80% augmented)
        use_augmented = random.random() < 0.8

        moved_any_image = False
        if use_augmented and 'augmented' in group:
            # Move the augmented image to val_images
            shutil.move(group['augmented'], val_images_dir)
            print(f"Moved augmented image: {group_name}_augmented.png")
            moved_any_image = True
            # The original image remains in the training set
        elif 'original' in group:
            # Fallback to original if augmented doesn't exist or not chosen
            shutil.move(group['original'], val_images_dir)
            print(f"Moved original image: {group_name}.jpg")
            moved_any_image = True
        
        # Always move the corresponding gt file if an image was moved
        if moved_any_image:
            if os.path.exists(gt_path):
                shutil.move(gt_path, val_gts_dir)
            else:
                print(f"CRITICAL WARNING: Could not find gt for moved image: {gt_base_name}")
        else:
            print(f"WARNING: No image found to move for group: {group_name}")

    # --- 4. Post-split verification and cleanup of the training set ---
    print("\n--- Step 4: Verifying and cleaning the training set against remaining gts ---")
    # This step is crucial to remove any training images (original or augmented) whose
    # corresponding gt was moved to the validation set.
    verify_and_clean(images_dir, gts_dir, '.jpg')
    verify_and_clean(augmented_dir, gts_dir, '_augmented.png')

    print("\n--- Dataset Preparation Summary ---")
    train_original_count = len(glob.glob(os.path.join(images_dir, '*.jpg')))
    train_augmented_count = len(glob.glob(os.path.join(augmented_dir, '*.png')))
    train_gts_count = len(glob.glob(os.path.join(gts_dir, '*.png')))
    val_image_count = len(glob.glob(os.path.join(val_images_dir, '*')))
    val_gts_count = len(glob.glob(os.path.join(val_gts_dir, '*.png')))

    print(f"Training data groups: {train_gts_count}")
    print(f"  - Original images in training set: {train_original_count}")
    print(f"  - Augmented images in training set: {train_augmented_count}")
    print(f"Validation data groups: {val_gts_count}")
    print(f"  - Images in validation set: {val_image_count}")
    print("\nDataset preparation complete.")


if __name__ == '__main__':
    dataset_base_path = './finetune_dataset'
    clean_and_prepare_dataset(dataset_base_path)