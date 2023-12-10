import yaml
import tqdm
import argparse
import os
import torch
import shutil
from torch.optim.lr_scheduler import MultiStepLR
from easydict import EasyDict

from tools.load import load_data
from tools.utils import setup_logger, to_cpu, plot_loss
from models.cdrnet import CDRNet
from models.loss import JointsMSESmoothLoss
from models.metrics import calc_mpjpe


def run(config):
    logger = setup_logger()

    model_path = os.path.join("weights", config.MODEL.NAME)
    if os.path.exists(model_path):
        while True:
            logger.warning("Model name exists, "
                           "do you want to override the previous model?")
            inp = input(">> [y:n]")
            if inp.lower()[0] == "y":
                shutil.rmtree(model_path)
                break
            elif inp.lower()[0] == "n":
                logger.info("Stop training!")
                exit(0)
    os.makedirs(model_path)

    train_dataset, valid_dataset, train_loader, valid_loader \
        = load_data(config)

    logger.info("The number of data in train set: {}"
                .format(train_dataset.__len__()))
    logger.info("The number of data in valid set: {}"
                .format(valid_dataset.__len__()))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info("Using device: {}".format(device))

    model = CDRNet(config)
    if len(config.MODEL.PRETRAINED) > 0:
        logger.info("Load pretrained weights from '{}'"
                    .format(config.MODEL.PRETRAINED))
        model.init_weights(config.MODEL.PRETRAINED)
    model = model.to(device)

    criterion = JointsMSESmoothLoss(config.LOSS.USE_TARGET_WEIGHT)

    optimizer = torch.optim.Adam(model.parameters(), config.TRAIN.LR)
    scheduler = MultiStepLR(
        optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR
    )

    val_best_error = float("inf")

    n_joints = 19
    base_joint = 1
    scale_3d = 0.1
    train_losses = []
    val_losses = []
    val_errors = []

    for epoch in range(config.TRAIN.EPOCH):
        train_loss, val_loss = 0, 0

        # -------------------
        # ------ Train ------
        # -------------------

        model.train()
        logger.info(('\n' + '%10s' * 4) % ('Epoch', 'lr', 'loss', 'grad_norm'))
        pbar = enumerate(train_loader)
        pbar = tqdm.tqdm(pbar, total=len(train_loader))
        for i, (image_left, image_right, target_3d,
                target_left, target_right, meta) in pbar:
            image_left = image_left.to(device)
            image_right = image_right.to(device)
            target_3d = target_3d.to(device)
            target_left = target_left.to(device)
            target_right = target_right.to(device)

            P_left = meta["P_left"].to(device)
            P_right = meta["P_right"].to(device)

            target_weight = meta["joints_vis"].to(device)

            imgs = [image_left, image_right]
            Ps = [P_left, P_right]
            targets = [target_left, target_right]

            optimizer.zero_grad()

            pred_2ds, pred_3ds = model(imgs, Ps)

            pred_3ds[:, torch.arange(n_joints) != base_joint] \
                -= pred_3ds[:, base_joint:base_joint+1]
            target_3d[:, torch.arange(n_joints) != base_joint] \
                -= target_3d[:, base_joint:base_joint+1]

            loss = torch.zeros(1, device=device)
            if epoch < config.TRAIN.WARMUP:
                for pred, target in zip(pred_2ds, targets):
                    loss += criterion(pred, target, target_weight)
            else:
                loss += config.TRAIN.LOSS_3D_WEIGHT * \
                    criterion(
                        pred_3ds * scale_3d, target_3d * scale_3d,
                        target_weight)
                loss_2d = 0
                for pred, target in zip(pred_2ds, targets):
                    loss_2d += criterion(pred, target, target_weight)

                loss += loss_2d
       
            loss.backward()

            grad_norm = torch.norm(
                torch.cat([p.grad.flatten() for p in model.parameters()]))

            if not epoch < config.TRAIN.WARMUP:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 100)

            optimizer.step()

            train_loss += loss.item()

            s = ('%10s' + '%10.4g' * 3) \
                % ('%g/%g' % (epoch + 1, config.TRAIN.EPOCH),
                   optimizer.param_groups[0]["lr"], loss, grad_norm)
            pbar.set_description(s)
            pbar.update(0)

        scheduler.step()

        # --------------------
        # ---- Validation ----
        # --------------------

        model.eval()
        error = 0
        with torch.no_grad():
            for i, (image_left, image_right, target_3d,
                    target_left, target_right, meta) in tqdm.tqdm(
                    enumerate(valid_loader), total=len(valid_loader)):
                image_left = image_left.to(device)
                image_right = image_right.to(device)
                target_3d = target_3d.to(device)
                target_left = target_left.to(device)
                target_right = target_right.to(device)

                P_left = meta["P_left"].to(device)
                P_right = meta["P_right"].to(device)

                target_weight = meta["joints_vis"].to(device)

                imgs = [image_left, image_right]
                Ps = [P_left, P_right]
                targets = [target_left, target_right]

                pred_2ds, pred_3ds = model(imgs, Ps)
                loss = torch.zeros(1, device=device)
                if epoch < config.TRAIN.WARMUP:
                    for pred, target in zip(pred_2ds, targets):
                        loss += criterion(pred, target, target_weight)
                else:
                    loss += config.TRAIN.LOSS_3D_WEIGHT * \
                        criterion(
                            pred_3ds * scale_3d, target_3d * scale_3d,
                            target_weight)
                    loss_2d = 0
                    for pred, target in zip(pred_2ds, targets):
                        loss_2d += criterion(pred, target, target_weight)

                    loss += loss_2d

                for i in range(2):
                    pred_2ds[i] = to_cpu(pred_2ds[i])
                pred_3ds = to_cpu(pred_3ds)
                target_3d = to_cpu(target_3d)
                target_left = to_cpu(target_left)
                target_right = to_cpu(target_right)
                target_weight = to_cpu(target_weight)

                err = calc_mpjpe(
                    pred_2ds, pred_3ds, target_3d,
                    target_left, target_right,
                    target_weight)

                error += err[1]  # 3d MPJPE

                val_loss += loss.item()

        train_loss /= train_loader.__len__()
        val_loss /= valid_loader.__len__()
        error /= valid_loader.__len__()
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_errors.append(error)
        # --------------------------
        # Logging Stage
        # --------------------------
        print("Epoch: ", epoch + 1)
        print("train_loss: {}"
              .format(train_loss))
        print("val_loss: {}"
              .format(val_loss))
        print("val_error: {}"
              .format(error))

        # save best model
        if error < val_best_error and epoch > config.TRAIN.WARMUP:  # the loss is identical to the metric
            val_best_error = error

            save_folder = os.path.join(model_path, "best.pth")
            torch.save(model.state_dict(), save_folder)
            logger.info("Current best model is saved!")

        # save latest model
        save_folder = os.path.join(model_path, "latest.pth")
        torch.save(model.state_dict(), save_folder)

    plot_loss(train_losses, "/plot", "Training Loss")
    plot_loss(val_losses, "/plot", "Validation Loss")
    plot_loss(val_errors, "/plot", "MPJPE")
    logger.info("Training is done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str,
                        default="configs/mads_3d.yaml",
                        help="Path to the config file")
    args = parser.parse_args()

    with open(args.config_path, 'r') as f:
        config = EasyDict(yaml.safe_load(f))

    run(config)
