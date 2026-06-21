import torch


def build_optimizer(model, cfg):
    name = cfg["optimizer"]["name"]

    params = [p for p in model.parameters() if p.requires_grad]

    if name == "sgd":
        return torch.optim.SGD(
            params,
            lr=cfg["optimizer"]["lr"],
            momentum=cfg["optimizer"]["momentum"],
            weight_decay=cfg["optimizer"]["weight_decay"],
        )

    if name == "adamw":
        return torch.optim.AdamW(
            params,
            lr=cfg["optimizer"]["lr"],
            weight_decay=cfg["optimizer"]["weight_decay"],
        )

    raise ValueError(f"Unknown optimizer: {name}")