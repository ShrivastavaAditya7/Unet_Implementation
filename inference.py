import torch
import torch.nn.functional as F
from torchvision.io import read_image
from PIL import Image
import numpy as np
from pathlib import Path
import argparse
import matplotlib.pyplot as plt

from main import UNet

def preprocess_image(path, img_size):
    img = read_image(str(path)).float() / 255.0
    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)
    img = img.unsqueeze(0)
    img = F.interpolate(img, size=(img_size, img_size), mode='bilinear', align_corners=False)
    return img.squeeze(0)

def postprocess_mask(mask_tensor, orig_size=None, threshold=0.5):
    if mask_tensor.ndim == 3:
        mask_tensor = mask_tensor.squeeze(0)
    mask_prob = torch.sigmoid(mask_tensor)
    mask_bin = (mask_prob >= threshold).cpu().numpy().astype(np.uint8) * 255
    if orig_size is not None:
        pil = Image.fromarray(mask_bin)
        pil = pil.resize(orig_size[::-1], resample=Image.NEAREST)
        return pil
    return Image.fromarray(mask_bin)

def overlay_mask_on_image(image_path, mask_pil, alpha=0.5):
    img = Image.open(image_path).convert("RGB")
    mask_arr = np.array(mask_pil.convert("L"))
    red = np.zeros((img.size[1], img.size[0], 4), dtype=np.uint8)
    red[mask_arr > 127] = [255, 0, 0, int(255 * alpha)]
    mask_overlay = Image.fromarray(red, mode="RGBA")
    out = Image.alpha_composite(img.convert("RGBA"), mask_overlay)
    return out.convert("RGB")

def save_concatenated_figure(orig_pil, mask_pil, overlay_pil, out_path):
    try:
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(orig_pil); axes[0].set_title("Input"); axes[0].axis('off')
        axes[1].imshow(np.array(mask_pil.convert('L')), cmap='gray'); axes[1].set_title("Mask"); axes[1].axis('off')
        axes[2].imshow(overlay_pil); axes[2].set_title("Overlay"); axes[2].axis('off')
        plt.tight_layout(); fig.savefig(str(out_path), bbox_inches='tight', dpi=150); plt.close(fig)
    except Exception:
        try: plt.close('all')
        except Exception: pass
        raise

def predict_single(model, image_path, img_size, device, threshold):
    model.eval()
    img = preprocess_image(image_path, img_size)
    orig = Image.open(image_path)
    orig_size = (orig.size[1], orig.size[0])
    with torch.no_grad():
        inp = img.unsqueeze(0).to(device)
        out = model(inp)
        out = out.cpu().squeeze(0)
    mask_pil = postprocess_mask(out, orig_size=orig_size, threshold=threshold)
    return mask_pil

def run_folder(model, input_dir, output_dir, img_size, device, threshold, overlay=False):
    input_dir = Path(input_dir); output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    for img_path in sorted(input_dir.glob("*.jpg")) + sorted(input_dir.glob("*.png")):
        mask_pil = predict_single(model, img_path, img_size, device, threshold)
        mask_out = output_dir / (img_path.stem + "_mask.png"); mask_pil.save(mask_out)
        if overlay:
            ov = overlay_mask_on_image(img_path, mask_pil); ov.save(output_dir / (img_path.stem + "_overlay.png"))
        try:
            orig = Image.open(img_path).convert("RGB")
            overlay_pil = ov if overlay else overlay_mask_on_image(img_path, mask_pil)
            concat_out = output_dir / (img_path.stem + "_concat.png")
            save_concatenated_figure(orig, mask_pil, overlay_pil, concat_out)
        except Exception as e:
            print(f"Warning: failed concat for {img_path}: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="best_unet_sa1b.pth")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="predictions")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--overlay", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = UNet(n_channels=3, n_classes=1)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.to(device); model.eval()

    inp = Path(args.input); out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    if inp.is_file():
        mask_pil = predict_single(model, inp, args.img_size, device, args.threshold)
        mask_pil.save(out / (inp.stem + "_mask.png"))
        if args.overlay:
            overlay = overlay_mask_on_image(inp, mask_pil); overlay.save(out / (inp.stem + "_overlay.png"))
        try:
            orig = Image.open(inp).convert("RGB")
            overlay_pil = overlay if args.overlay else overlay_mask_on_image(inp, mask_pil)
            save_concatenated_figure(orig, mask_pil, overlay_pil, out / (inp.stem + "_concat.png"))
        except Exception as e:
            print(f"Warning: failed concat for {inp}: {e}")
        print("Saved:", out)
    elif inp.is_dir():
        run_folder(model, inp, out, args.img_size, device, args.threshold, overlay=args.overlay)
        print("Saved predictions to", out)
    else:
        raise RuntimeError("Input path not found")

if __name__ == "__main__":
    main()