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
    Rotate90d,
)
from monai.handlers.utils import from_engine
from monai.inferers import sliding_window_inference
from monai.data import CacheDataset, DataLoader, Dataset, decollate_batch
import os
import sys
import torch
import numpy as np
import csv
import json
import cv2
from collections import OrderedDict
import logging
from data_preprocess import data_preprocess_for_inference
from train import setup_transforms, setup_model

current_dir = os.path.dirname(os.path.abspath(__file__))
monai_dir = os.path.dirname(current_dir)

sys.path.append(os.path.join(monai_dir, "config"))
from config import read_config_yaml, write_config_yaml

sys.path.append(os.path.join(monai_dir, "utils"))
from utils import os_makedirs, shutil_rmtree
from logger import get_logger

# Read config_file
config_file = os.path.join(monai_dir, 'config/config.yaml')
config = read_config_yaml(config_file)

# Get logger
# logging level (NOTSET=0 ; DEBUG=10 ; INFO=20 ; WARNING=30 ; ERROR=40 ; CRITICAL=50)
logger = get_logger(name=__file__, console_handler_level=logging.DEBUG, file_handler_level=None)


def contours_to_json(organ, contours, png_file_name, image_shape, json_file_path):
    json_dict = OrderedDict()

    if len(contours) > 0:
        for idx, contour in enumerate(contours):
            if idx==0:
                json_dict["version"] = "3.16.7"
                json_dict["flags"] = dict()
                json_dict["shapes"] = list()
                # json_dict["lineColor"] = [0, 255, 0, 128]
                # json_dict["fillColor"] = [255, 0, 0, 128]
                json_dict["imagePath"] = png_file_name
                json_dict["imageData"] = None
                json_dict["imageHeight"] = image_shape[0]
                json_dict["imageWidth"] = image_shape[1]

            shapes = OrderedDict()
            shapes["label"] = organ
            shapes["points"] = np.squeeze(np.array(contour)).tolist()
            shapes["shape_type"] = "polygon"
            shapes["flags"] = dict()
            json_dict["shapes"].append(shapes)

        with open(json_file_path, 'w') as json_file:
            json.dump(json_dict, json_file, sort_keys=False, indent=2, separators=(',', ': '))


if __name__ == "__main__":
    data_dir_list = ['/home/aoi/opencv_practice/monai/CT_organ_nckuh/004', 
                     '/home/aoi/opencv_practice/monai/CT_organ_nckuh/003',
                     '/home/aoi/opencv_practice/monai/CT_organ_nckuh/005']

    # # # # # # # # # # # # # # #
    #   Global configuration    #
    # # # # # # # # # # # # # # #
    # 1: liver / 3: pancreas / 4: spleen / 5: kidney
    organ = config['organ']
    roi_size = config['roi_size']
    channel_num = config['organ_to_mmar'][organ]['channel']

    # # # # # # # # # # # # #
    #   Data preprocess     #
    # # # # # # # # # # # # #
    imagesTs_dir = os.path.join(monai_dir, 'imagesTs')
    slicesTs_dir = os.path.join(monai_dir, 'slicesTs')

    scale_min, scale_max, imagesTs_path_list, slicesTs_path_list = \
        data_preprocess_for_inference(data_dir_list, imagesTs_dir, slicesTs_dir)
    print(scale_min, scale_max)
    print(imagesTs_path_list)
    print(slicesTs_path_list)


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #   Setup dataset path, transforms, CacheDataset and DataLoader for validation    #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    val_files = [
        {"image": image_name}
        for image_name in imagesTs_path_list
    ]

    val_transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            # Spacingd(keys=["image"], pixdim=(
            #     1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
            Orientationd(keys=["image"], axcodes="RAS"),
            # Rotate90d(keys=["image"], spatial_axes=(0,1)),
            # Flipd(keys=["image"], spatial_axis=1),
            ScaleIntensityRanged(
                keys=["image"], a_min=scale_min, a_max=scale_max,
                b_min=0.0, b_max=1.0, clip=True,
            ),
            CropForegroundd(keys=["image"], source_key="image"),
            EnsureTyped(keys=["image"]),
        ]
    )

    val_ds = CacheDataset(
        data=val_files, transform=val_transforms,
        cache_rate=1.0, num_workers=4)

    val_loader = DataLoader(val_ds, batch_size=1, num_workers=4)

    post_transforms = Compose([
        EnsureTyped(keys="pred"),
        Invertd(
            keys="pred",
            transform=val_transforms,
            orig_keys="image",
            meta_keys="pred_meta_dict",
            orig_meta_keys="image_meta_dict",
            meta_key_postfix="meta_dict",
            nearest_interp=False,
            to_tensor=True,
        ),
        Rotate90d(keys=["pred"], spatial_axes=(0,1)),
        AsDiscreted(keys="pred", argmax=True, to_onehot=channel_num)
    ])

    # # # # # # # # # # #
    #   Create Model    #
    # # # # # # # # # # #
    device, mmar_dir, model = setup_model(organ)
    model_file_path = os.path.join(mmar_dir, "best_metric_model.pth")
    model.load_state_dict(torch.load(model_file_path))


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #   Check best model output with the input image and label  #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    labelsTs_dir = os.path.join(monai_dir, 'labelsTs')
    os_makedirs(labelsTs_dir, keep_exists=True)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))

    model.eval()
    with torch.no_grad():
        for val_data, slicesTs_path in zip(val_loader, slicesTs_path_list):
            # 
            csv_file_name = os.path.basename(slicesTs_path)
            data_dir_name, ext = os.path.splitext(csv_file_name)
            slicesTs_list = []
            with open(slicesTs_path, 'r') as csvfile:
                rows = csv.reader(csvfile)
                for row in rows:
                    slicesTs_list.append(row)
            labelsTs_data_dir = os.path.join(labelsTs_dir, data_dir_name)
            os_makedirs(labelsTs_data_dir)


            # 
            val_inputs = val_data["image"].to(device)
            sw_batch_size = 4
            val_data["pred"] = sliding_window_inference(
                val_inputs, roi_size, sw_batch_size, model
            )
            post_val_data = [post_transforms(i) for i in decollate_batch(val_data)]
            val_outputs = from_engine(["pred"])(post_val_data)

            output = torch.argmax(val_outputs[0], dim=0).detach().cpu().numpy()
            num_slices = output.shape[2]
            for idx in range(num_slices):
                image = output[:,:,idx].astype('uint8')

                if np.any(image>0):
                    slice_location, dcm_file_name = slicesTs_list[idx]
                    png_file_name = dcm_file_name.replace('.dcm', '.png')
                    json_file_name = dcm_file_name.replace('.dcm', '.json')
                    json_file_path = os.path.join(labelsTs_data_dir, json_file_name)
                    print(idx, slice_location, dcm_file_name, json_file_path)

                    ret, thresh = cv2.threshold(image, 0, 255, 0)
                    image_bgr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
                    erosion = cv2.erode(image_bgr, kernel, iterations = 2)
                    dilation = cv2.dilate(erosion, kernel, iterations = 2)

                    image_gray = cv2.cvtColor(dilation, cv2.COLOR_BGR2GRAY)
                    contours, heirarchy = cv2.findContours(image_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    contours_to_json(organ, contours, png_file_name, image.shape, json_file_path)
