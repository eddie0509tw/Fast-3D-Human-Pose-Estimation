import yaml
import tqdm
import argparse
import os
import torch
import shutil
from torch.optim.lr_scheduler import MultiStepLR
from easydict import EasyDict

from tools.load import load_data
from tools.utils import setup_logger, plot_loss
from models.poseresnet import PoseResNet
from models.loss import JointsMSELoss, MPJPELoss, JointsMSESmoothLoss
from models.metrics import accuracy


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
    else:
        os.makedirs(model_path)

    train_dataset, valid_dataset, train_loader, valid_loader \
        = load_data(config)

    logger.info("The number of data in train set: {}"
                .format(train_dataset.__len__()))
    logger.info("The number of data in valid set: {}"
                .format(valid_dataset.__len__()))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info("Using device: {}".format(device))

    model = PoseResNet(config)
    if len(config.MODEL.PRETRAINED) > 0:
        logger.info("Load pretrained weights from '{}'"
                    .format(config.MODEL.PRETRAINED))
        model.init_weights(config.MODEL.PRETRAINED)
    model = model.to(device)

    if config.LOSS.TYPE == "MPJPE":
        criterion = MPJPELoss(config.LOSS.USE_TARGET_WEIGHT)
    elif config.LOSS.TYPE == "JointsMSESmooth":
        criterion = JointsMSESmoothLoss(config.LOSS.USE_TARGET_WEIGHT)
    elif config.LOSS.TYPE == "JointsMSE":
        criterion = JointsMSELoss(config.LOSS.USE_TARGET_WEIGHT)
    else:
        raise NotImplementedError

    optimizer = torch.optim.Adam(model.parameters(), config.TRAIN.LR)
    scheduler = MultiStepLR(
        optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR
    )

    best_acc = 0
    train_losses = []
    val_losses = []
    train_acc = []
    val_acc = []
    for epoch in range(config.TRAIN.EPOCH):
        train_loss, val_loss = 0, 0
        train_class_acc, val_class_acc = 0, 0

        # -------------------
        # ------ Train ------
        # -------------------

        model.train()
        logger.info(('\n' + '%10s' * 3) % ('Epoch', 'lr', 'loss'))
        pbar = enumerate(train_loader)
        pbar = tqdm.tqdm(pbar, total=len(train_loader))
        for i, (img, target, target_weight, meta) in pbar:
            img = img.to(device)
            target = target.to(device)
            target_weight = target_weight.to(device)

            optimizer.zero_grad()

            out = model(img)
            loss = criterion(out, target, target_weight)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            acc, _ = accuracy(out.detach().cpu().numpy(),
                              target.detach().cpu().numpy())
            train_class_acc += acc[0]

            s = ('%10s' + '%10.4g' * 2) \
                % ('%g/%g' % (epoch + 1, config.TRAIN.EPOCH),
                   optimizer.param_groups[0]["lr"], loss)
            pbar.set_description(s)
            pbar.update(0)

        scheduler.step()

        # --------------------
        # ---- Validation ----
        # --------------------

        model.eval()
        with torch.no_grad():
            for i, (img, target, target_weight, meta) in tqdm.tqdm(
                    enumerate(valid_loader), total=len(valid_loader)):
                img = img.to(device)
                target = target.to(device)
                target_weight = target_weight.to(device)

                out = model(img)
                loss = criterion(out, target, target_weight)

                val_loss += loss.item()

                acc, _ = accuracy(out.detach().cpu().numpy(),
                                  target.detach().cpu().numpy())
                val_class_acc += acc[0]
        # loss summarizing
        train_loss /= train_loader.__len__()
        val_loss /= valid_loader.__len__()
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_acc.append(train_class_acc / train_loader.__len__())
        val_acc.append(val_class_acc / valid_loader.__len__())
        # --------------------------
        # Logging Stage
        # --------------------------
        print("Epoch: ", epoch + 1)
        print("train_loss: {}, train_class_acc: {}"
              .format(train_loss / train_loader.__len__(),
                      train_class_acc / train_loader.__len__()))
        print("val_loss: {}, val_class_acc: {}"
              .format(val_loss / valid_loader.__len__(),
                      val_class_acc / valid_loader.__len__()))

        # save best model
        if val_class_acc > best_acc:
            best_acc = val_class_acc

            save_folder = os.path.join(model_path, "best.pth")
            torch.save(model.state_dict(), save_folder)
            logger.info("Current best model is saved!")

        # save latest model
        save_folder = os.path.join(model_path, "latest.pth")
        torch.save(model.state_dict(), save_folder)
    # plot_loss(train_losses, "./plot", "Training Loss")
    # plot_loss(val_losses, "./plot", "Validation Loss")
    # plot_loss(train_acc, "./plot", "Training Accuracy")
    # plot_loss(val_acc, "./plot", "Validation Accuracy")
    logger.info("Training is done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str,
                        default="configs/mads_2d.yaml",
                        help="Path to the config file")
    args = parser.parse_args()

    with open(args.config_path, 'r') as f:
        config = EasyDict(yaml.safe_load(f))

    run(config)
