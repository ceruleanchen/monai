from monai.utils import first, set_determinism
from monai.transforms import (
    AsDiscrete,
    AsDiscreted,
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRanged,
    Spacingd,
    EnsureTyped,
    EnsureType,
    Invertd,
)
from monai.handlers.utils import from_engine
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.metrics import DiceMetric
from monai.losses import DiceLoss
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
from monai.config import print_config
from monai.apps import download_and_extract, load_from_mmar
import torch
import matplotlib
import matplotlib.pyplot as plt
import tempfile
import shutil
import os
import sys
import json
import glob
import logging
import random
import math
import csv
import numpy as np
import argparse
from multiprocessing import Process, Lock

current_dir = os.path.dirname(os.path.abspath(__file__))
monai_dir = os.path.dirname(current_dir)

sys.path.append(os.path.join(monai_dir, "config"))
from config import read_config_yaml, write_config_yaml, write_config_yaml_with_key_value

sys.path.append(os.path.join(monai_dir, "utils"))
from utils import os_makedirs, shutil_rmtree
from logger import get_logger

# Read config_file
config_file = os.path.join(monai_dir, 'config/config.yaml')
config = read_config_yaml(config_file)
models_dir = config['models_dir']

# Get logger
# logging level (NOTSET=0 ; DEBUG=10 ; INFO=20 ; WARNING=30 ; ERROR=40 ; CRITICAL=50)
logger = get_logger(name=__file__, console_handler_level=logging.INFO, file_handler_level=None)


def setup_dataset_path(organ, train_data_dir):
    train_images = sorted(
        glob.glob(os.path.join(train_data_dir, "imagesTr", "*.nii.gz")))
    train_labels = sorted(
        glob.glob(os.path.join(train_data_dir, "labelsTr", "*_{}.nii.gz".format(organ))))
    data_dicts = [
        {"image": image_name, "label": label_name}
        for image_name, label_name in zip(train_images, train_labels)
    ]
    return data_dicts

def setup_transforms(roi_size):
    scale_min = config['scale_min']
    scale_max = config['scale_max']

    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            # Spacingd(keys=["image", "label"], pixdim=(
            #     1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            # Rotate90d(keys=["image", "label"], spatial_axes=(0,1)),
            # Flipd(keys=["image", "label"], spatial_axis=1),
            ScaleIntensityRanged(
                keys=["image"], a_min=scale_min, a_max=scale_max,
                b_min=0.0, b_max=1.0, clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=roi_size,
                pos=1,
                neg=1,
                num_samples=4,
                image_key="image",
                image_threshold=0,
            ),
            # user can also add other random transforms
            # RandAffined(
            #     keys=['image', 'label'],
            #     mode=('bilinear', 'nearest'),
            #     prob=1.0, spatial_size=(96, 96, 96),
            #     rotate_range=(0, 0, np.pi/15),
            #     scale_range=(0.1, 0.1, 0.1)),
            EnsureTyped(keys=["image", "label"]),
        ]
    )

    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            # Spacingd(keys=["image", "label"], pixdim=(
            #     1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            # Rotate90d(keys=["image", "label"], spatial_axes=(0,1)),
            # Flipd(keys=["image", "label"], spatial_axis=1),
            ScaleIntensityRanged(
                keys=["image"], a_min=scale_min, a_max=scale_max,
                b_min=0.0, b_max=1.0, clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            EnsureTyped(keys=["image", "label"]),
        ]
    )
    return train_transforms, val_transforms

def setup_train_ds_and_loader(train_files, train_transforms):
    train_ds = CacheDataset(
        data=train_files, transform=train_transforms,
        cache_rate=1.0, num_workers=4)
    # train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)

    # use batch_size=2 to load images and use RandCropByPosNegLabeld
    # to generate 2 x 4 images for network training
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=4)
    return train_ds, train_loader

def setup_val_ds_and_loader(val_files, val_transforms):
    val_ds = CacheDataset(
        data=val_files, transform=val_transforms,
        cache_rate=1.0, num_workers=4)
    # val_ds = Dataset(data=val_files, transform=val_transforms)

    val_loader = DataLoader(val_ds, batch_size=1, num_workers=4)
    return val_ds, val_loader

def setup_model(organ, gpu_num=0):
    mmar_dir = config['organ_to_mmar'][organ]['mmar_dir']
    os_makedirs(mmar_dir, keep_exists=True)

    if config['production'] == 'retrain_aifs' or config['production'] == 'inference_aifs':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda:{}".format(gpu_num) if torch.cuda.is_available() else "cpu")
    unet_model = load_from_mmar(
        config['organ_to_mmar'][organ]['name'], mmar_dir=mmar_dir,
        map_location=device, pretrained=True)
    model = unet_model.to(device)
    return device, model

def validation_process(lock, input_dict):
    organ = input_dict['organ']
    channel_num = input_dict['channel_num']
    roi_size = input_dict['roi_size']
    val_loader = input_dict['val_loader']
    device = input_dict['device']
    model = input_dict['model']
    dice_metric = input_dict['dice_metric']

    post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=channel_num)])
    post_label = Compose([EnsureType(), AsDiscrete(to_onehot=channel_num)])
    model.eval()
    with torch.no_grad():
        for val_data in val_loader:
            val_inputs, val_labels = (
                val_data["image"].to(device),
                val_data["label"].to(device),
            )
            sw_batch_size = 2
            val_outputs = sliding_window_inference(
                val_inputs, roi_size, sw_batch_size, model)
            val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
            val_labels = [post_label(i) for i in decollate_batch(val_labels)]
            # compute metric for current iteration
            dice_metric(y_pred=val_outputs, y=val_labels)

        # aggregate the final mean dice result
        metric = dice_metric.aggregate().item()
        # reset the status for next validation round
        dice_metric.reset()

    round_metric = round(metric, 4)

    lock.acquire()
    config = read_config_yaml(config_file)
    config['organ_to_mmar'][organ]['old_model_dice'] = round_metric
    write_config_yaml(config_file, config)
    lock.release()

def training_process(lock, input_dict):
    organ = input_dict['organ']
    channel_num = input_dict['channel_num']
    roi_size = input_dict['roi_size']
    train_ds = input_dict['train_ds']
    train_loader = input_dict['train_loader']
    val_loader = input_dict['val_loader']
    device = input_dict['device']
    model = input_dict['model']
    loss_function = input_dict['loss_function']
    optimizer = input_dict['optimizer']
    dice_metric = input_dict['dice_metric']
    new_model_dir = input_dict['new_model_dir']
    max_epochs = input_dict['max_epochs']

    os_makedirs(new_model_dir)
    date = os.path.basename(os.path.dirname(new_model_dir))
    new_model_file_name = "{}_{}.pth".format(organ, date)
    new_model_file_path = os.path.join(new_model_dir, new_model_file_name)

    val_interval = 2
    best_metric = -1
    best_metric_epoch = -1
    # epoch_loss_values = []
    # metric_values = []
    post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=channel_num)])
    post_label = Compose([EnsureType(), AsDiscrete(to_onehot=channel_num)])

    metric_csv_path = os.path.join(new_model_dir, 'metric.csv')
    with open(metric_csv_path, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['epoch', 'train_loss', 'val_dice'])
        metric_list = []
        for epoch in range(max_epochs):
            metric_ele_list = []
            metric_ele_list.append(epoch+1)
            # training
            logger.debug("-" * 10)
            logger.debug(f"epoch {epoch + 1}/{max_epochs}")
            model.train()
            epoch_loss = 0
            step = 0
            for batch_data in train_loader:
                step += 1
                inputs, labels = (
                    batch_data["image"].to(device),
                    batch_data["label"].to(device),
                )
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = loss_function(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                logger.debug(
                    f"{step}/{math.ceil(len(train_ds) / train_loader.batch_size)}, "
                    f"train_loss: {loss.item():.4f}")
            epoch_loss /= step
            # epoch_loss_values.append(epoch_loss)
            round_epoch_loss = round(epoch_loss, 4)
            metric_ele_list.append(round_epoch_loss)
            logger.debug(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

            # validation
            if (epoch + 1) % val_interval == 0:
                model.eval()
                with torch.no_grad():
                    for val_data in val_loader:
                        val_inputs, val_labels = (
                            val_data["image"].to(device),
                            val_data["label"].to(device),
                        )
                        sw_batch_size = 2
                        val_outputs = sliding_window_inference(
                            val_inputs, roi_size, sw_batch_size, model)
                        val_outputs = [post_pred(i) for i in decollate_batch(val_outputs)]
                        val_labels = [post_label(i) for i in decollate_batch(val_labels)]
                        # compute metric for current iteration
                        dice_metric(y_pred=val_outputs, y=val_labels)

                    # aggregate the final mean dice result
                    metric = dice_metric.aggregate().item()
                    # reset the status for next validation round
                    dice_metric.reset()

                    # metric_values.append(metric)
                    round_metric = round(metric, 4)
                    metric_ele_list.append(round_metric)
                    if metric > best_metric:
                        best_metric = metric
                        best_metric_epoch = epoch + 1
                        torch.save(model.state_dict(), new_model_file_path)

                        lock.acquire()
                        config = read_config_yaml(config_file)
                        config['organ_to_mmar'][organ]['new_model_file_path'] = new_model_file_path
                        config['organ_to_mmar'][organ]['new_model_dice'] = round_metric
                        write_config_yaml(config_file, config)
                        lock.release()
                        logger.debug("saved new best metric model")

                    logger.debug(f"current epoch: {epoch + 1}    current mean dice: {metric:.4f}")
                    logger.debug(f"best mean dice: {best_metric:.4f} at epoch: {best_metric_epoch}")
            else:
                metric_ele_list.append('')

            metric_list.append(metric_ele_list)
            if len(metric_list) >= 10:
                writer.writerows(metric_list)
                metric_list = []

        logger.debug(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
        if len(metric_list) > 0:
            writer.writerows(metric_list)
            metric_list = []
    return metric_csv_path

def metric_csv_to_png(metric_csv_path):
    metric_png_path = metric_csv_path.replace("csv", "png")
    with open(metric_csv_path, 'r') as csvfile:
        rows = csv.reader(csvfile)
        rows_array = np.array(list(rows))

        train_loss_epoch = [int(element) for element in rows_array[1:,0]]
        train_loss = [round(float(element),5) for element in rows_array[1:,1]]

        val_dice = rows_array[1:,2]
        val_dice_idx = np.where(val_dice!='')
        val_dice_epoch = [int(element) for element in rows_array[1:,0][val_dice_idx]]
        val_dice = [round(float(element),5) for element in val_dice[val_dice_idx]]

        matplotlib.use('Agg')
        plt.figure("train", (12, 6))
        plt.subplot(1, 2, 1)
        plt.title("Train Loss")
        plt.xlabel("epoch")
        plt.plot(train_loss_epoch, train_loss)
        plt.subplot(1, 2, 2)
        plt.title("Validation Dice")
        plt.xlabel("epoch")
        plt.plot(val_dice_epoch, val_dice)
        plt.savefig(metric_png_path)

def training(lock, organ, gpu_num=0):
    # # # # # # # # # # # # # # #
    #   Global configuration    #
    # # # # # # # # # # # # # # #
    # print_config()
    # 1: liver / 3: pancreas / 4: spleen / 5: kidney
    train_data_dir = config['train_data_dir']
    val_data_dir = config['val_data_dir']
    old_model_file_path = config['organ_to_mmar'][organ]['old_model_file_path']
    roi_size = config['roi_size']

    # # # # # # # # # # # # #
    #   Setup dataset path  #
    # # # # # # # # # # # # #
    train_files = setup_dataset_path(organ, train_data_dir)
    val_files = setup_dataset_path(organ, val_data_dir)


    # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #   Setup transforms for training and validation    #
    # # # # # # # # # # # # # # # # # # # # # # # # # # #
    train_transforms, val_transforms = setup_transforms(roi_size)


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #   Define CacheDataset and DataLoader for training and validation  #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    train_ds, train_loader = setup_train_ds_and_loader(train_files, train_transforms)
    val_ds, val_loader = setup_val_ds_and_loader(val_files, val_transforms)


    # # # # # # # # # # # # # # # # # # #
    #   Create Model, Loss, Optimizer   #
    # # # # # # # # # # # # # # # # # # #
    device, model = setup_model(organ, gpu_num=gpu_num)
    if old_model_file_path != None and os.path.isfile(old_model_file_path):
        logger.info("Load {} model from {}".format(organ, old_model_file_path))
        model.load_state_dict(torch.load(old_model_file_path))
    else:
        logger.info("Load default {} model from MMAR".format(organ))

    loss_function = DiceLoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.Adam(model.parameters(), 1e-4)
    dice_metric = DiceMetric(include_background=False, reduction="mean")


    # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #   Execute a typical PyTorch validation process    #
    # # # # # # # # # # # # # # # # # # # # # # # # # # #
    input_dict = {
                    'organ': organ,
                    'channel_num': config['organ_to_mmar'][organ]['channel'],
                    'roi_size': roi_size,
                    'val_loader': val_loader,
                    'device': device,
                    'model': model,
                    'dice_metric': dice_metric
                 }
    validation_process(lock, input_dict)


    # # # # # # # # # # # # # # # # # # # # # # # # #
    #   Execute a typical PyTorch training process  #
    # # # # # # # # # # # # # # # # # # # # # # # # #
    input_dict = {
                    'organ': organ,
                    'channel_num': config['organ_to_mmar'][organ]['channel'],
                    'roi_size': roi_size,
                    'train_ds': train_ds,
                    'train_loader': train_loader,
                    'val_loader': val_loader,
                    'device': device,
                    'model': model,
                    'loss_function': loss_function,
                    'optimizer': optimizer,
                    'dice_metric': dice_metric,
                    'new_model_dir': config['organ_to_mmar'][organ]['new_model_dir'],
                    'max_epochs': os.getenv('MAX_EPOCHS', 200)
                 }
    metric_csv_path = training_process(lock, input_dict)


    # # # # # # # # # # # # # # # # #
    #   Convert metric csv to png   #
    # # # # # # # # # # # # # # # # #
    metric_csv_to_png(metric_csv_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()


    supported_organ_list = list(config['organ_to_mmar'].keys())
    organ_list = os.getenv('ORGAN_LIST', supported_organ_list)
    if type(organ_list)==str:
        organ_list = json.loads(organ_list)

    for organ in list(organ_list):
        if organ not in supported_organ_list:
            organ_list.remove(organ)
    config['organ_list'] = organ_list
    write_config_yaml_with_key_value(config_file, 'organ_list', organ_list)
    logger.info("Effective organ_list is {}. (Support only {})".format(organ_list, supported_organ_list))


    lock = Lock()
    proc_list = []
    for organ in organ_list:
        if args.debug:
            while logger.hasHandlers():
                logger.removeHandler(logger.handlers[0])
            models_dir = config['models_dir']
            log_file_path = os.path.join(models_dir, "{}.log".format(organ))
            logger = get_logger(name=__file__, console_handler_level=None, file_handler_level=logging.INFO, file_name=log_file_path)

        proc = Process(target=training, args=(lock, organ, 0))
        proc.start()
        proc_list.append(proc)

        if args.debug:
            while logger.hasHandlers():
                logger.removeHandler(logger.handlers[0])
            logger = get_logger(name=__file__, console_handler_level=logging.INFO, file_handler_level=None)
        logger.info("Training on {} starts.".format(organ))

    for proc in proc_list:
        proc.join()
