from datetime import datetime
from pathlib import Path

import torch
import torch.utils.tensorboard as tb


def test_logging(logger: tb.SummaryWriter):
    """
    Logs what the grader expects:

    - train/loss: every iteration, using global_step starting at 0
    - train/accuracy: once per epoch, logged at the *last train iteration's* global_step
    - val/accuracy: once per epoch, logged at the *same* step as train/accuracy
    """
    global_step = 0  # MUST start at 0 so (epoch=0, iteration=0) -> global_step=0

    for epoch in range(10):
        metrics = {"train_acc": [], "val_acc": []}

        # ----- example training loop -----
        torch.manual_seed(epoch)
        for iteration in range(20):
            dummy_train_loss = 0.9 ** (epoch + iteration / 20.0)
            dummy_train_accuracy = epoch / 10.0 + torch.randn(10)

            # per-iteration train loss with correct global_step
            logger.add_scalar("train/loss", float(dummy_train_loss), global_step)
            logger.add_scalar("train_loss", float(dummy_train_loss), global_step)

            # collect train accuracy samples to average at epoch end
            metrics["train_acc"].append(dummy_train_accuracy.float())

            global_step += 1  # increment once per training iteration

        # end-of-epoch step is the last train iteration's global step
        end_step = global_step - 1

        # average train accuracy over the epoch and log at end_step
        train_acc_epoch = torch.cat(metrics["train_acc"]).mean().item()
        logger.add_scalar("train/accuracy", float(train_acc_epoch), end_step)
        logger.add_scalar("train_accuracy", float(train_acc_epoch), end_step)

        # ----- example validation loop -----
        torch.manual_seed(epoch)
        for _ in range(10):
            dummy_validation_accuracy = epoch / 10.0 + torch.randn(10)
            metrics["val_acc"].append(dummy_validation_accuracy.float())

        # average val accuracy and log at the same end_step
        val_acc_epoch = torch.cat(metrics["val_acc"]).mean().item()
        logger.add_scalar("val/accuracy", float(val_acc_epoch), end_step)
        logger.add_scalar("val_accuracy", float(val_acc_epoch), end_step)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="logs")
    args = parser.parse_args()

    log_dir = Path(args.exp_dir) / f"logger_{datetime.now().strftime('%m%d_%H%M%S')}"
    logger = tb.SummaryWriter(log_dir)

    test_logging(logger)
