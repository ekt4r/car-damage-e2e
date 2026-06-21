import torch


def build_scheduler(optimizer, cfg, epochs):
    name = cfg["scheduler"]["name"]

    if name is None or name == "none":
        return None

    if name == "cosine":
        t_max = cfg["scheduler"].get("t_max") or epochs
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=t_max,
        )

    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg["scheduler"]["step_size"],
            gamma=cfg["scheduler"]["gamma"],
        )

    raise ValueError(f"Unknown scheduler: {name}")