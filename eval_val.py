import torch
import numpy as np
from main import UNet, SA1BDataset
from torch.utils.data import DataLoader, random_split
from pathlib import Path

def dice_score(pred, gt, eps=1e-6):
    pred_f = pred.astype(np.uint8)
    gt_f = gt.astype(np.uint8)
    inter = (pred_f & gt_f).sum()
    return 2 * inter / (pred_f.sum() + gt_f.sum() + eps)

def iou_score(pred, gt, eps=1e-6):
    pred_f = pred.astype(np.uint8)
    gt_f = gt.astype(np.uint8)
    inter = (pred_f & gt_f).sum()
    union = (pred_f | gt_f).sum()
    return inter / (union + eps)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(n_channels=3, n_classes=1)
    model.load_state_dict(torch.load('best_unet_sa1b.pth', map_location=device))
    model.to(device).eval()

    full = SA1BDataset("sa1b_dataset", transform=None)
    val_split = 0.2
    train_size = int((1 - val_split) * len(full))
    val_size = len(full) - train_size
    _, val_ds = random_split(full, [train_size, val_size])

    dl = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    dices, ious = [], []
    for item in dl:
        img = item['image'][0].numpy()
        gt = item['mask'][0].numpy()
        # TODO: reuse inference preprocessing and model forward to get mask_pred
        pass

    # print averages after implementing inference loop