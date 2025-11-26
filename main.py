import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import torchvision.transforms.v2 as v2
from torchvision.io import read_image
from pycocotools import mask as mask_utils
import json
from pathlib import Path
import tqdm
import numpy as np
from PIL import Image
import random

# Dataset and mask handling
class ImageMaskResize(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.size = size

    def __call__(self, image: torch.Tensor, mask: torch.Tensor) -> tuple:
        image = v2.functional.resize(
            image,
            (self.size, self.size),
            interpolation=v2.functional.InterpolationMode.BILINEAR,
        )
        # Resize mask with nearest neighbor for binary
        if len(mask.shape) == 3:  # If multi-mask, union them
            mask = torch.any(mask, dim=0).float().unsqueeze(0)
        mask = v2.functional.resize(
            mask,
            (self.size, self.size),
            interpolation=v2.functional.InterpolationMode.NEAREST_EXACT,
        ).float()
        return image, mask


class ImageMaskTransform:
    # module-level picklable transform wrapper
    def __init__(self, size: int):
        self.resize = ImageMaskResize(size)
        # Determine a ToImage-like transform if available
        try:
            ToImageTransform = getattr(v2, 'ToImage')
        except Exception:
            ToImageTransform = getattr(v2, 'ToImagePIL', None)

        if ToImageTransform is None:
            self.to_image = None
        else:
            # Instantiate the transform (it should be picklable as it's defined at module level)
            self.to_image = ToImageTransform()

    def __call__(self, image: torch.Tensor, mask: torch.Tensor):
        # avoid unnecessary PIL conversions
        if self.to_image is not None and not isinstance(image, torch.Tensor):
            image = self.to_image(image)
        # If a PIL image, convert back to tensor
        if isinstance(image, Image.Image):
            arr = np.array(image)
            # arr shape: (H, W) or (H, W, C)
            if arr.ndim == 2:
                arr = np.expand_dims(arr, axis=2)
            # Ensure channel-last to channel-first
            arr = arr.astype(np.float32) / 255.0
            arr = torch.from_numpy(arr).permute(2, 0, 1)
            image = arr

        return self.resize(image, mask)

class SA1BDataset(torch.utils.data.Dataset):
    def __init__(self, data_folder: str, transform=None):
        super().__init__()
        self.data_folder = Path(data_folder)
        
        # Glob and sort numerically by the number in 'sa_N.jpg'
        def numerical_key(path):
            return int(path.stem.split('_')[1])
        
        self.images_file_paths = sorted(
            list(self.data_folder.glob("sa_*.jpg")), key=numerical_key
        )
        self.json_file_paths = sorted(
            list(self.data_folder.glob("sa_*.json")), key=numerical_key
        )
        
        assert len(self.images_file_paths) == len(self.json_file_paths), "Mismatch in image/JSON count"
        print(f"Loaded {len(self.images_file_paths)} paired samples from {data_folder}")
        
        self.transform = transform

    def __len__(self):
        return len(self.images_file_paths)

    def __getitem__(self, index):
    # load image
        image_path = self.images_file_paths[index]
        # read_image accepts a file path and returns a uint8 tensor (C,H,W)
        image = read_image(str(image_path)).float() / 255.0  # Normalize to [0,1]

        # Load annotations
        json_path = self.json_file_paths[index]
        with open(json_path) as f:
            data = json.load(f)
            annotations = data.get("annotations", [])


        # decode annotations to numpy masks and union
        masks = []
        for annot in annotations:
            if "segmentation" in annot:
                rle = annot["segmentation"]
                if isinstance(rle, dict) and "counts" in rle:  # RLE format
                    mask_np = mask_utils.decode(rle)
                else:  # Polygon format (list of lists)
                    mask_np = mask_utils.decode({
                        "size": [data["image"]["height"], data["image"]["width"]],
                        "counts": rle,
                    })
                # ensure uint8 (0/1)
                mask_np = mask_np.astype(np.uint8)
                masks.append(mask_np)

        img_h, img_w = data["image"]["height"], data["image"]["width"]
        if masks:
            # Union masks in-place to avoid creating a big (N,H,W) array.
            union_np = masks[0]
            for m in masks[1:]:
                union_np |= m
            union_mask = torch.from_numpy(union_np).unsqueeze(0).float()
            del masks
        else:
            # No annotations -> empty mask
            union_mask = torch.zeros((1, img_h, img_w), dtype=torch.float)

        if self.transform:
            image, union_mask = self.transform(image, union_mask)

        # Target: binary mask (1 channel, 0/1)
        target = (union_mask > 0).float()

        return {"image": image, "mask": target}

# Model: small U-Net
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_ch, enc_ch, out_ch):
        super().__init__()
        # after transposed conv we halve the channels of x1
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        # conv input channels = up_out_channels + enc_ch
        self.conv = DoubleConv(in_ch // 2 + enc_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Pad if dimensions don't match
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        # channel bookkeeping
        c1, c2, c3, c4 = 64, 128, 256, 512
        c5 = 1024 // factor
        # Up(in_ch, enc_ch, out_ch)
        self.up1 = Up(c5, c4, c4 // factor)
        self.up2 = Up(c4 // factor, c3, c3 // factor)
        self.up3 = Up(c3 // factor, c2, c2 // factor)
        self.up4 = Up(c2 // factor, c1, c1)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

def train_model(model, train_loader, val_loader, num_epochs=10, lr=1e-3, device='cuda',
                checkpoint_path: str = 'checkpoint.pth', resume_checkpoint: str = None,
                checkpoint_interval_batches: int = 200):
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')

    # Resume state (if we load a checkpoint later)
    start_epoch = 0
    resume_batch = 0

    def save_checkpoint(path, epoch, batch_idx):
        ckpt = {
            'epoch': epoch,
            'batch': batch_idx,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'torch_rng': torch.get_rng_state(),
            'np_rng': np.random.get_state(),
            'py_rng': random.getstate(),
        }
        if torch.cuda.is_available():
            try:
                ckpt['cuda_rng'] = torch.cuda.get_rng_state_all()
            except Exception:
                pass
        torch.save(ckpt, path)

    def load_checkpoint(path):
        nonlocal start_epoch, resume_batch, best_val_loss
        ckpt = torch.load(path, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        try:
            optimizer.load_state_dict(ckpt.get('optimizer_state', {}))
        except Exception:
            pass
        best_val_loss = ckpt.get('best_val_loss', best_val_loss)
        start_epoch = ckpt.get('epoch', 0)
        resume_batch = ckpt.get('batch', 0)
        # restore RNGs (best-effort)
        try:
            torch.set_rng_state(ckpt['torch_rng'])
        except Exception:
            pass
        try:
            np.random.set_state(ckpt['np_rng'])
        except Exception:
            pass
        try:
            random.setstate(ckpt['py_rng'])
        except Exception:
            pass
        if torch.cuda.is_available() and 'cuda_rng' in ckpt:
            try:
                torch.cuda.set_rng_state_all(ckpt['cuda_rng'])
            except Exception:
                pass

    if resume_checkpoint is not None:
        print(f"Resuming from checkpoint: {resume_checkpoint}")
        load_checkpoint(resume_checkpoint)

    for epoch in range(start_epoch, num_epochs):
    # train
        model.train()
        train_loss = 0.0
        # If resuming mid-epoch, advance iterator
        if epoch == start_epoch and resume_batch > 0:
            train_iter = iter(train_loader)
            for _ in range(resume_batch):
                try:
                    next(train_iter)
                except StopIteration:
                    break
            train_pbar = tqdm.tqdm(train_iter, total=len(train_loader), initial=resume_batch, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
            batch_start_idx = resume_batch
        else:
            train_pbar = tqdm.tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
            batch_start_idx = 0

        for batch_idx, batch in enumerate(train_pbar, start=batch_start_idx):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix({'loss': train_loss / (train_pbar.n + 1)})

            # Periodic checkpoint (save epoch and batch index so we can resume mid-epoch)
            if (batch_idx + 1) % checkpoint_interval_batches == 0:
                save_checkpoint(checkpoint_path, epoch, batch_idx + 1)
        
    # validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            val_pbar = tqdm.tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for batch in val_pbar:
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
                
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                val_pbar.set_postfix({'loss': val_loss / (val_pbar.n + 1)})
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_unet_sa1b.pth')

    return model

# Step 4: Usage Example
if __name__ == "__main__":
    data_folder = "sa1b_dataset"
    img_size = 256
    batch_size = 8
    val_split = 0.2
    num_epochs = 20

    # picklable transform
    transform = ImageMaskTransform(img_size)
    full_dataset = SA1BDataset(data_folder, transform=transform)
    train_size = int((1 - val_split) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = UNet(n_channels=3, n_classes=1)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    resume_checkpoint = None

    trained_model = train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=num_epochs,
        device=device,
        checkpoint_path='checkpoint.pth',
        resume_checkpoint=resume_checkpoint,
        checkpoint_interval_batches=200,
    )

    torch.save(trained_model.state_dict(), 'final_unet_sa1b.pth')
    print("Training complete! Best model saved as 'best_unet_sa1b.pth'")