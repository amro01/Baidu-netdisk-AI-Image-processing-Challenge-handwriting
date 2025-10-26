import os
from os import listdir
from os.path import join, isfile
import paddle
import numpy as np
import cv2
from os import walk
from os.path import join
import random
from PIL import Image
import os

from paddle.vision.transforms import Compose, RandomCrop, ToTensor
from paddle.vision.transforms import functional as F

# --- Helper functions from original dataloader ---

def random_horizontal_flip(imgs):
    if random.random() < 0.3:
        for i in range(len(imgs)):
            imgs[i] = imgs[i].transpose(Image.FLIP_LEFT_RIGHT)
    return imgs

def random_rotate(imgs):
    if random.random() < 0.3:
        max_angle = 10
        angle = random.random() * 2 * max_angle - max_angle
        for i in range(len(imgs)):
            img = np.array(imgs[i])
            w, h = img.shape[:2]
            rotation_matrix = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1)
            img_rotation = cv2.warpAffine(img, rotation_matrix, (h, w))
            imgs[i] = Image.fromarray(img_rotation)
    return imgs

def CheckImageFile(filename):
    return any(filename.endswith(extention) for extention in ['.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG', '.bmp', '.BMP'])

def ImageTransform():
    return Compose([
        ToTensor(),
    ])

# --- Mixed Data Dataloader ---

class MixedErasingData(paddle.io.Dataset):
    """
    A dataset class that mixes original and augmented images for training.
    It loads augmented images with a certain probability and original images otherwise.
    """
    def __init__(self, dataRoot, augmentedRoot, gtsRoot, maskRoot, loadSize, training=True, mask_dir='mask', augmented_prob=0.8):
        super(MixedErasingData, self).__init__()
        
        # List of original and augmented image files
        self.originalFiles = sorted([join(dataRoot, f) for f in listdir(dataRoot) if CheckImageFile(f)])
        self.augmentedFiles = sorted([join(augmentedRoot, f) for f in listdir(augmentedRoot) if CheckImageFile(f)])
        
        print(f"Found {len(self.originalFiles)} original images.")
        print(f"Found {len(self.augmentedFiles)} augmented images.")

        # Directly use the provided root paths
        self.gtsRoot = gtsRoot
        self.maskRoot = maskRoot
        
        self.loadSize = loadSize
        self.ImgTrans = ImageTransform()
        self.training = training
        self.RandomCropparam = RandomCrop(self.loadSize)
        self.augmented_prob = augmented_prob

    def __getitem__(self, index):
        # Decide whether to use an augmented or original image based on probability
        use_augmented = random.random() < self.augmented_prob
        
        try:
            if use_augmented and self.augmentedFiles:
                # --- Use augmented image ---
                img_path = self.augmentedFiles[index % len(self.augmentedFiles)]
                base_filename = os.path.basename(img_path)
                # GT path: .../gts/dehw_train_xxxx.png
                gt_filename = base_filename.replace('_augmented.png', '.png')
                gt_path = os.path.join(self.gtsRoot, gt_filename)
                # Mask path: .../mask/dehw_train_xxxx_augmented.png
                mask_path = os.path.join(self.maskRoot, base_filename)
            elif self.originalFiles:
                # --- Use original image ---
                img_path = self.originalFiles[index % len(self.originalFiles)]
                base_filename = os.path.basename(img_path)
                # GT path: .../gts/dehw_train_xxxx.png
                gt_filename = base_filename.replace('.jpg', '.png')
                gt_path = os.path.join(self.gtsRoot, gt_filename)
                # Mask path: .../mask/dehw_train_xxxx.png
                mask_path = os.path.join(self.maskRoot, gt_filename)
            else:
                 raise IndexError("No files to select from.")

            # Load images
            img = Image.open(img_path).convert('RGB')
            mask = Image.open(mask_path).convert('RGB')
            gt = Image.open(gt_path).convert('RGB')

        except FileNotFoundError as e:
            print(f"Error loading files for index {index}: {e}")
            # Handle error by loading the next available valid item
            return self.__getitem__((index + 1) % self.__len__())
            
        # Apply data augmentation (flips, rotations)
        if self.training:
            all_input = [img, mask, gt]
            all_input = random_horizontal_flip(all_input)   
            all_input = random_rotate(all_input)
            img, mask, gt = all_input[0], all_input[1], all_input[2]
            
        # Apply paired random crop
        param = self.RandomCropparam._get_param(img, self.loadSize)
        inputImage = F.crop(img, *param)
        maskIn = F.crop(mask, *param)
        groundTruth = F.crop(gt, *param)
        
        # Convert to tensor
        inputImage = self.ImgTrans(inputImage)
        maskIn = self.ImgTrans(maskIn)
        groundTruth = self.ImgTrans(groundTruth)
        
        # Return filename for debugging
        path = img_path.split('/')[-1]
        
        return inputImage, groundTruth, maskIn, path
    
    def __len__(self):
        # The length of the dataset is the sum of both original and augmented files
        # to ensure the dataloader iterates enough times to sample from both pools.
        return len(self.originalFiles) + len(self.augmentedFiles)


# New Dataloader for mixed validation set
class MixedDevData(paddle.io.Dataset):
    def __init__(self, dataRoot, gtRoot):
        super(MixedDevData, self).__init__()
        self.imageFiles = [join(dataRoot, f) for f in listdir(dataRoot) if CheckImageFile(f)]
        self.gtRoot = gtRoot
        self.ImgTrans = ImageTransform()

    def __getitem__(self, index):
        # Load the validation image
        img_path = self.imageFiles[index]
        img = Image.open(img_path).convert('RGB')
        
        # Determine the corresponding GT path
        base_name = os.path.basename(img_path)
        if '_augmented' in base_name:
            # For augmented images like 'dehw_train_xxxx_augmented.png'
            # the GT is 'dehw_train_xxxx.png'
            gt_base_name = base_name.replace('_augmented', '')
        else:
            # For original images like 'dehw_train_xxxx.jpg'
            # the GT is 'dehw_train_xxxx.png'
            gt_base_name = os.path.splitext(base_name)[0] + '.png'
            
        gt_path = join(self.gtRoot, gt_base_name)
        gt = Image.open(gt_path).convert('RGB')

        # Apply transformations
        inputImage = self.ImgTrans(img)
        groundTruth = self.ImgTrans(gt)
        
        return inputImage, groundTruth, os.path.basename(img_path)

    def __len__(self):
        return len(self.imageFiles)
