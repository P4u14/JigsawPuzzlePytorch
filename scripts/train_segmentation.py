#!/usr/bin/env python3
"""
Train a segmentation head on top of the pretrained JigsawNetwork.
Usage:
    python scripts/train_segmentation.py configs/seg_config.yaml
"""
import os
import sys
# ensure project root is on the import path
script_dir   = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, os.pardir))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import glob
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image, ImageDraw
import numpy as np

# import your jigsaw backbone
from JigsawNetwork import Network as JigsawModel
# import the segmentation dataset & head you created
from data.segmentation_dataset import SegmentationDataset
from model.segmentation_head import SegmentationHead


def load_encoder(checkpoint_path):
    # Instantiate the same architecture used during pretraining
    jigsaw = JigsawModel(classes=1000)
    # Load pretrained weights (excludes the final fc8)
    jigsaw.load(checkpoint_path)
    # Keep only the convolutional backbone
    encoder = jigsaw.conv
    return encoder


def main(config_path):
    # Load configuration
    cfg = yaml.safe_load(open(config_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare output directory
    output_dir = cfg["checkpoint"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # --- Prepare training dataset & loader ---
    # Construct paths from config
    base_path = os.path.expanduser(cfg["io"]["downstream_base_path"])
    train_data_path = os.path.join(base_path, cfg["io"]["train_path"])

    # Find all image files and filter out masks
    all_train_files = sorted(glob.glob(os.path.join(train_data_path, cfg["io"]["images_glob"])))
    train_imgs = [f for f in all_train_files if "-mask" not in os.path.basename(f)]

    # Find all mask files
    train_masks = sorted(glob.glob(os.path.join(train_data_path, cfg["io"]["masks_glob"])))

    train_ds = SegmentationDataset(
        train_imgs,
        train_masks,
        img_size=tuple(cfg.get("img_size", (256,256)))
    )
    train_dl = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=4
    )

    # --- Optional validation ---
    val_dl = None
    if cfg["io"].get("val_path"):
        val_data_path = os.path.join(base_path, cfg["io"]["val_path"])
        all_val_files = sorted(glob.glob(os.path.join(val_data_path, cfg["io"]["images_glob"])))
        val_imgs = [f for f in all_val_files if "-mask" not in os.path.basename(f)]
        val_masks = sorted(glob.glob(os.path.join(val_data_path, cfg["io"]["masks_glob"])))

        val_ds = SegmentationDataset(
            val_imgs,
            val_masks,
            img_size=tuple(cfg.get("img_size", (256,256)))
        )
        val_dl = DataLoader(
            val_ds,
            batch_size=cfg["training"]["batch_size"],
            shuffle=False,
            num_workers=4
        )

    # Build model: encoder + segmentation head
    encoder = load_encoder(cfg["checkpoint"]["pretrained_ckpt"]).to(device)
    head = SegmentationHead(
        in_channels=cfg["model"]["in_channels"],
        num_classes=cfg["model"]["num_classes"],
        upsample_factor=cfg["model"]["upsample_factor"]
    ).to(device)
    model = nn.Sequential(encoder, head).to(device)

    # Freeze encoder for initial training
    for p in encoder.parameters():
        p.requires_grad = False

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["training"]["lr_head"]
    )
    criterion = nn.BCEWithLogitsLoss()
    dice_coeff = lambda p, t: (2 * (p * t).sum()) / (p.sum() + t.sum() + 1e-6)

    # Training loop
    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        total_loss = 0.0
        total_dice = 0.0
        for imgs, masks in train_dl:
            imgs = imgs.to(device)
            masks = masks.to(device)

            logits = model(imgs)
            # The model outputs a different size than the masks, so we need to resize it
            # before computing the loss.
            logits = F.interpolate(
                logits,
                size=masks.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
            loss = criterion(logits, masks)

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            dice = dice_coeff(preds, masks).item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            total_dice += dice * imgs.size(0)
        avg_loss = total_loss / len(train_dl.dataset)
        avg_dice = total_dice / len(train_dl.dataset)
        print(f"Epoch {epoch+1}/{cfg['training']['epochs']} - Train Loss: {avg_loss:.4f}, Dice: {avg_dice:.4f}")

        # Validation step
        if val_dl:
            model.eval()
            val_loss = 0.0
            val_dice = 0.0
            with torch.no_grad():
                for imgs, masks in val_dl:
                    imgs = imgs.to(device)
                    masks = masks.to(device)
                    logits = model(imgs)
                    logits = F.interpolate(
                        logits,
                        size=masks.shape[-2:],
                        mode='bilinear',
                        align_corners=False
                    )
                    loss = criterion(logits, masks)

                    probs = torch.sigmoid(logits)
                    preds = (probs > 0.5).float()
                    dice = dice_coeff(preds, masks).item()

                    val_loss += loss.item() * imgs.size(0)
                    val_dice += dice * imgs.size(0)
            avg_val_loss = val_loss / len(val_dl.dataset)
            avg_val_dice = val_dice / len(val_dl.dataset)
            print(f"Epoch {epoch+1}/{cfg['training']['epochs']} - Val Loss:   {avg_val_loss:.4f}, Dice: {avg_val_dice:.4f}")

            # Create a directory for the current epoch's visualizations
            vis_dir = os.path.join(output_dir, "visualizations", f"epoch_{epoch+1:03d}")
            os.makedirs(vis_dir, exist_ok=True)

            # Denormalize images for visualization
            # NOTE: These values are standard for ImageNet. Adjust if your data uses different stats.
            inv_normalize = transforms.Normalize(
                mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                std=[1/0.229, 1/0.224, 1/0.225]
            )

            # --- Visualize all validation images ---
            model.eval()
            val_img_counter = 0
            with torch.no_grad():
                for imgs, masks in val_dl:
                    imgs = imgs.to(device)
                    masks = masks.to(device)
                    logits = model(imgs)
                    logits = F.interpolate(
                        logits,
                        size=masks.shape[-2:],
                        mode='bilinear',
                        align_corners=False
                    )
                    probs = torch.sigmoid(logits)
                    preds = (probs > 0.5).float()

                    for img_tensor, mask_tensor, pred_tensor in zip(imgs.cpu(), masks.cpu(), preds.cpu()):
                        # Denormalize and convert image to PIL
                        img_tensor_denorm = inv_normalize(img_tensor)
                        img_pil = transforms.ToPILImage()(img_tensor_denorm)
                        draw = ImageDraw.Draw(img_pil)

                        # Convert masks to numpy and find contours
                        mask_np = mask_tensor.squeeze().numpy().astype(np.uint8)
                        pred_np = pred_tensor.squeeze().numpy().astype(np.uint8)

                        # Find contours using a simple method (checking for non-zero neighbors)
                        def find_contours(mask):
                            contours = []
                            for y in range(1, mask.shape[0] - 1):
                                for x in range(1, mask.shape[1] - 1):
                                    if mask[y, x] > 0:
                                        # Check if it's a border pixel
                                        if (mask[y-1, x] == 0 or mask[y+1, x] == 0 or
                                            mask[y, x-1] == 0 or mask[y, x+1] == 0):
                                            contours.append((x, y))
                            return contours

                        # Draw ground truth contours (pink)
                        gt_contours = find_contours(mask_np)
                        if gt_contours:
                            draw.point(gt_contours, fill=(255, 105, 180)) # Pink

                        # Draw prediction contours (orange)
                        pred_contours = find_contours(pred_np)
                        if pred_contours:
                            draw.point(pred_contours, fill=(255, 165, 0)) # Orange

                        # Save the visualized image
                        save_image(transforms.ToTensor()(img_pil), os.path.join(vis_dir, f"val_img_{val_img_counter:04d}.png"))
                        val_img_counter += 1


        # Unfreeze encoder after specified epochs
        if (epoch+1) == cfg["training"].get("freeze_epochs", 5):
            for p in encoder.parameters():
                p.requires_grad = True
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["lr_finetune"])
            print("Unfroze encoder; now fine-tuning entire model.")

        # Save checkpoint
        ckpt = {
            "epoch": epoch+1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        ckpt_name = f"segmentation_epoch{epoch+1:03d}.pth"
        torch.save(ckpt, os.path.join(output_dir, ckpt_name))

if __name__ == "__main__":
    cfg_file = sys.argv[1] if len(sys.argv) > 1 else "configs/seg_config.yaml"
    main(cfg_file)
