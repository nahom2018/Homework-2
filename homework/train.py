import argparse
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.utils.tensorboard as tb

from .models import ClassificationLoss, load_model, save_model
from .utils import load_data

# Try to import a factory that builds a model from a name.
# If your starter code calls this something else (e.g., make_model/get_model),
# adjust the import below accordingly.
try:
    from .models import build_model  # preferred name
except Exception:  # pragma: no cover
    try:
        from .models import make_model as build_model  # common alias
    except Exception:
        build_model = None


def train(
    exp_dir: str = "logs",
    model_name: str = "linear",
    num_epoch: int = 50,
    lr: float = 1e-3,
    batch_size: int = 128,
    seed: int = 2024,
    ckpt: str | None = None,   # <-- optional checkpoint path (file), not a directory
    **kwargs,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        print("CUDA/MPS not available, using CPU")
        device = torch.device("cpu")

    # set random seed so each run is deterministic
    torch.manual_seed(seed)
    np.random.seed(seed)

    # directory with timestamp to save tensorboard logs and model checkpoints
    log_dir = Path(exp_dir) / f"{model_name}_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    # ---------------------------------------------------------------------
    # Build the model first, then (optionally) load weights from a checkpoint
    # ---------------------------------------------------------------------
    if build_model is None:
        raise RuntimeError(
            "Could not find a model factory (expected models.build_model or models.make_model). "
            "Import the correct constructor from homework.models and rename it here."
        )

    # kwargs may contain extra hyperparameters for the model constructor
    model = build_model(model_name, **kwargs)

    # If a checkpoint file is provided, load weights into the constructed model
    if ckpt is not None:
        if os.path.isfile(ckpt):
            load_model(model, ckpt, map_location="cpu")
            print(f"Loaded checkpoint: {ckpt}")
        else:
            raise FileNotFoundError(
                f"ckpt='{ckpt}' is not a file. Pass a valid .pt/.pth path or omit --ckpt to train from scratch."
            )

    model = model.to(device)
    model.train()

    # NOTE: these are dataset *directories*. Do not pass these to torch.load.
    train_data = load_data("classification_data/train", shuffle=True, batch_size=batch_size, num_workers=2)
    val_data = load_data("classification_data/val", shuffle=False)

    # create loss function and optimizer
    loss_func = ClassificationLoss()
    # optimizer = ...

    global_step = 0
    metrics = {"train_acc": [], "val_acc": []}

    # training loop
    for epoch in range(num_epoch):
        # clear metrics at beginning of epoch
        for key in metrics:
            metrics[key].clear()

        model.train()

        for img, label in train_data:
            img, label = img.to(device), label.to(device)

            # TODO: implement training step
            raise NotImplementedError("Training step not implemented")

            global_step += 1

        # disable gradient computation and switch to evaluation mode
        with torch.inference_mode():
            model.eval()

            for img, label in val_data:
                img, label = img.to(device), label.to(device)

                # TODO: compute validation accuracy
                raise NotImplementedError("Validation accuracy not implemented")

        # log average train and val accuracy to tensorboard
        epoch_train_acc = torch.as_tensor(metrics["train_acc"]).mean()
        epoch_val_acc = torch.as_tensor(metrics["val_acc"]).mean()

        # TODO: actually log to TensorBoard
        raise NotImplementedError("Logging not implemented")
        # Example when you implement:
        # logger.add_scalar("acc/train", float(epoch_train_acc), epoch)
        # logger.add_scalar("acc/val", float(epoch_val_acc), epoch)

        # print on first, last, every 10th epoch
        if epoch == 0 or epoch == num_epoch - 1 or (epoch + 1) % 10 == 0:
            print(
                f"Epoch {epoch + 1:2d} / {num_epoch:2d}: "
                f"train_acc={epoch_train_acc:.4f} "
                f"val_acc={epoch_val_acc:.4f}"
            )

    # save and overwrite the model in the root directory for grading
    save_model(model)

    # save a copy of model weights in the log directory
    torch.save(model.state_dict(), log_dir / f"{model_name}.th")
    print(f"Model saved to {log_dir / f'{model_name}.th'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--exp_dir", type=str, default="logs")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--num_epoch", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--ckpt", type=str, default=None, help="Optional checkpoint file to resume from")

    # optional: additional model hyperparameters (forwarded via **kwargs)
    # parser.add_argument("--num_layers", type=int, default=3)

    # pass all arguments to train
    train(**vars(parser.parse_args()))
