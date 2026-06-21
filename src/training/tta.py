import torch


@torch.no_grad()
def predict_with_tta(model, images, enabled=False):
    if not enabled:
        return model(images)

    outputs_original = model(images)

    flipped_images = [torch.flip(img, dims=[2]) for img in images]
    outputs_flipped = model(flipped_images)

    merged_outputs = []

    for image, out_orig, out_flip in zip(images, outputs_original, outputs_flipped):
        width = image.shape[2]

        boxes_flip = out_flip["boxes"].clone()
        x1 = boxes_flip[:, 0].clone()
        x2 = boxes_flip[:, 2].clone()

        boxes_flip[:, 0] = width - x2
        boxes_flip[:, 2] = width - x1

        boxes = torch.cat([out_orig["boxes"], boxes_flip], dim=0)
        scores = torch.cat([out_orig["scores"], out_flip["scores"]], dim=0)
        labels = torch.cat([out_orig["labels"], out_flip["labels"]], dim=0)

        merged_outputs.append(
            {
                "boxes": boxes,
                "scores": scores,
                "labels": labels,
            }
        )

    return merged_outputs