import os
import argparse
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.io import DataLoader
from data.mixed_dataloader import MixedErasingData, MixedDevData # Import the new mixed dataloaders
from loss.Loss import LossWithGAN_STE
from models.sa_gan import STRnet2
from models.sa_aidr import STRAIDR
import utils
import random
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--numOfWorkers', type=int, default=16, help='workers for dataloader')
parser.add_argument('--modelsSavePath', type=str, default='ckpts_finetune', help='path for saving models')
parser.add_argument('--logPath', type=str, default='logs_finetune')
parser.add_argument('--batchSize', type=int, default=4)
parser.add_argument('--loadSize', type=int, default=512, help='image loading size')
# Add arguments for original and augmented data roots
parser.add_argument('--dataRoot', type=str, default='./finetune_dataset/images', help='path to original images')
parser.add_argument('--augmentedRoot', type=str, default='./finetune_dataset/augmented_images', help='path to augmented images')
parser.add_argument('--pretrained',type=str, default='', help='pretrained models for finetuning')
parser.add_argument('--num_epochs', type=int, default=1000, help='epochs')
parser.add_argument('--net', type=str, default='str')
parser.add_argument('--lr', type=float, default=5e-5)
parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decay factor for step decay')
parser.add_argument('--lr_decay_iters', type=int, default=100000, help='learning rate decay per N iters')
parser.add_argument('--mask_dir', type=str, default='mask')
parser.add_argument('--seed', type=int, default=2022)
# Add argument for augmentation probability
parser.add_argument('--augmented_prob', type=float, default=0.8, help='probability of using augmented images')
parser.add_argument('--accumulation_steps', type=int, default=1, help='gradient accumulation steps')
args = parser.parse_args()

# Use a specific log file for finetuning
log_file = os.path.join(args.logPath, args.net + '_finetune_log.txt')
if not os.path.exists(args.logPath):
    os.makedirs(args.logPath)
logging = utils.setup_logger(output=log_file, name=args.net + '_finetune')
logging.info(args)

# set gpu
if paddle.is_compiled_with_cuda():
    paddle.set_device('gpu:0')
else:
    paddle.set_device('cpu')

# set random seed
logging.info('========> Random Seed: {}'.format(args.seed))
random.seed(args.seed)
np.random.seed(args.seed)
paddle.seed(args.seed)
paddle.framework.random._manual_program_seed(args.seed)


batchSize = args.batchSize
loadSize = (args.loadSize, args.loadSize)

if not os.path.exists(args.modelsSavePath):
    os.makedirs(args.modelsSavePath)

# --- Use the new MixedErasingData dataloader ---
gtsRoot = os.path.join(os.path.dirname(args.dataRoot), 'gts')
maskRoot = os.path.join(os.path.dirname(args.dataRoot), args.mask_dir)
Erase_data = MixedErasingData(
    dataRoot=args.dataRoot,
    augmentedRoot=args.augmentedRoot,
    gtsRoot=gtsRoot,
    maskRoot=maskRoot,
    loadSize=loadSize,
    training=True,
    mask_dir=args.mask_dir,
    augmented_prob=args.augmented_prob
)
Erase_data = DataLoader(Erase_data, batch_size=batchSize, shuffle=True, num_workers=args.numOfWorkers, drop_last=True)
logging.info(f"Training data loaded: {len(Erase_data.dataset)} samples.")
# -----------------------------------------------

# --- Use the new MixedDevData for validation ---
val_dataRoot = './finetune_dataset/val_images'
val_gtRoot = './finetune_dataset/val_gts'
Erase_val_data = MixedDevData(dataRoot=val_dataRoot, gtRoot=val_gtRoot)
Erase_val_data = DataLoader(Erase_val_data, batch_size=1, shuffle=False, num_workers=4, drop_last=False)
logging.info(f"Validation data loaded with MixedDevData: {len(Erase_val_data.dataset)} samples.")
# -------------------------------------------

logging.info('==============> Net to be used: {}'.format(args.net))
if args.net == 'str':
    netG = STRnet2(3)
elif args.net == 'idr':
    netG = STRAIDR(num_c=96)

if args.pretrained:
    logging.info('Loading pretrained weights from: {}'.format(args.pretrained))
    weights = paddle.load(args.pretrained)
    netG.load_dict(weights)
    logging.info('Pretrained weights loaded successfully.')

count = 1
scheduler = paddle.optimizer.lr.StepDecay(learning_rate=args.lr, step_size=args.lr_decay_iters, gamma=args.gamma, verbose=False)
G_optimizer = paddle.optimizer.Adam(scheduler, parameters=netG.parameters(), weight_decay=0.0)

criterion = LossWithGAN_STE(lr=0.00001, betasInit=(0.0, 0.9), Lamda=10.0)
logging.info('Optimizer and loss function initialized.')

num_epochs = args.num_epochs
best_psnr = 0
iters = 0
for epoch in range(1, num_epochs + 1):
    netG.train()

    for k, (imgs, gt, masks, path) in enumerate(Erase_data):
        iters += 1
        
        x_o1, x_o2, x_o3, fake_images, mm = netG(imgs)
        G_loss = criterion(imgs, masks, x_o1, x_o2, x_o3, fake_images, mm, gt, count, epoch)
        G_loss = G_loss.sum()

        # Normalize loss for gradient accumulation
        if args.accumulation_steps > 1:
            G_loss = G_loss / args.accumulation_steps
        
        G_loss.backward()

        # Update weights only after accumulating gradients
        if iters % args.accumulation_steps == 0:
            G_optimizer.step()
            scheduler.step()
            G_optimizer.clear_grad()
        if iters % 100 == 0:
            logging.info('[{}/{}] Generator Loss of epoch {} is {:.5f}, Lr:{:.6f}'.format(iters, len(Erase_data) * num_epochs, epoch, G_loss.item(), G_optimizer.get_lr()))
        count += 1
    
        if (iters % 5000 == 0):
            netG.eval()
            val_psnr = 0
            logging.info("Starting validation...")
            for index, (imgs_val, gt_val, path_val) in enumerate(Erase_val_data):
                _,_,h,w = imgs_val.shape
                rh, rw = h, w
                step = 512
                pad_h = step - h if h < step else 0
                pad_w = step - w if w < step else 0
                m = nn.Pad2D((0, pad_w,0, pad_h))
                imgs_val = m(imgs_val)
                _, _, h, w = imgs_val.shape
                res = paddle.zeros_like(imgs_val)
                for i in range(0, h, step):
                    for j in range(0, w, step):
                        if h - i < step: i = h - step
                        if w - j < step: j = w - step
                        clip = imgs_val[:, :, i:i+step, j:j+step]
                        with paddle.no_grad():
                            _, _, _, g_images_clip, mm = netG(clip)
                        
                        mm = paddle.where(F.sigmoid(mm)>0.5, paddle.zeros_like(mm), paddle.ones_like(mm))
                        g_image_clip_with_mask = clip * (mm) + g_images_clip * (1- mm)
                        res[:, :, i:i+step, j:j+step] = g_image_clip_with_mask
                
                res = res[:, :, :rh, :rw]
                output = utils.pd_tensor2img(res)
                target = utils.pd_tensor2img(gt_val)
                psnr = utils.compute_psnr(target, output)
                val_psnr += psnr
                # logging.info('Validation index:{} psnr: {}'.format(index, psnr)) # This can be too verbose

            ave_psnr = val_psnr/(index+1)
            model_save_path = os.path.join(args.modelsSavePath, 'STE_{}_iter_{}_{:.4f}.pdparams'.format(args.net, iters, ave_psnr))
            paddle.save(netG.state_dict(), model_save_path)
            logging.info(f"Model saved to {model_save_path}")

            if ave_psnr > best_psnr:
                best_psnr = ave_psnr
                best_model_path = os.path.join(args.modelsSavePath, 'STE_{}_best.pdparams'.format(args.net))
                paddle.save(netG.state_dict(), best_model_path)
                logging.info(f"New best model saved to {best_model_path}")

            logging.info('Epoch: {}, Iter: {}, Avg PSNR: {:.4f}, Best PSNR: {:.4f}'.format(epoch, iters, ave_psnr, best_psnr))
            netG.train() # Switch back to training mode