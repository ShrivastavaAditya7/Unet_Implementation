#!/usr/bin/env python
"""Run DETR object detector to find a person bbox, then use SAM to segment the person.
Saves output to `getty_person_output.png` in the workspace.
"""
from transformers import pipeline, SamImageProcessor, SamModel
from PIL import Image
import torch
import numpy as np

IMG = 'gettyimages-2208183148-612x612.jpg'

def find_person_bbox(detections):
    # Try to find a detection whose label contains 'person'
    for det in detections:
        label = str(det.get('label', '')).lower()
        if 'person' in label:
            # Try common box representations
            box = det.get('box') or det.get('bbox') or det.get('bbox_xyxy') or det.get('bbox_xywh')
            if isinstance(box, dict):
                # dict with keys like xmin/xmax or x/y/width/height
                if 'xmin' in box and 'xmax' in box:
                    x1 = int(box['xmin'])
                    y1 = int(box['ymin'])
                    x2 = int(box['xmax'])
                    y2 = int(box['ymax'])
                    return (x1, y1, x2, y2)
                # some pipelines return xywh list
            if isinstance(box, (list, tuple)):
                if len(box) == 4:
                    # Try interpreting as [xmin, ymin, xmax, ymax]
                    x1, y1, x2, y2 = box
                    # If values look like [x, y, w, h] (w small), convert
                    if x2 <= 1.0 or y2 <= 1.0:
                        # probably normalized coords - skip
                        continue
                    return (int(x1), int(y1), int(x2), int(y2))
    return None


def main():
    print('Loading image...')
    img = Image.open(IMG).convert('RGB')
    print(f'Image size: {img.size}')

    print('Loading DETR detector (this may take a moment)...')
    detector = pipeline('object-detection', model='facebook/detr-resnet-50', device=-1)
    print('Running detector...')
    dets = detector(img)
    print(f'DETECTIONS: {dets}')

    bbox = find_person_bbox(dets)
    if bbox is None:
        print('No person bbox found by DETR. Exiting.')
        return

    x1, y1, x2, y2 = bbox
    print(f'Person bbox (xyxy): {x1},{y1},{x2},{y2}')

    print('Loading SAM model...')
    processor = SamImageProcessor.from_pretrained('facebook/sam-vit-base')
    model = SamModel.from_pretrained('facebook/sam-vit-base')
    model.eval()

    # Prepare input
    inputs = processor(img, return_tensors='pt')
    input_boxes = torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32)

    print('Running SAM inference...')
    with torch.no_grad():
        outputs = model(pixel_values=inputs['pixel_values'], input_boxes=input_boxes.unsqueeze(0))

    masks = processor.post_process_masks(
        outputs.pred_masks,
        inputs['original_sizes'],
        inputs['reshaped_input_sizes']
    )

    if len(masks) == 0 or len(masks[0]) == 0:
        print('SAM did not produce masks')
        return

    best_mask = masks[0][0][0].cpu().numpy()
    if best_mask.dtype == bool:
        best_mask = best_mask.astype(np.uint8) * 255

    arr = np.array(img)
    overlay = arr.copy().astype(np.float32)
    mask_bool = best_mask > 0
    print(f'Segmented pixels: {np.sum(mask_bool)}')

    if np.sum(mask_bool) > 0:
        overlay[mask_bool] = overlay[mask_bool] * 0.6 + np.array([255, 0, 0]) * 0.4
        result_img = Image.fromarray(overlay.astype(np.uint8))
        result_img.save('getty_person_output.png')
        print('Saved getty_person_output.png')
    else:
        print('Mask contained no positive pixels')


if __name__ == '__main__':
    main()
