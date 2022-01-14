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
test_data_dir = config['test_data_dir']
models_dir = config['models_dir']

# Get logger
# logging level (NOTSET=0 ; DEBUG=10 ; INFO=20 ; WARNING=30 ; ERROR=40 ; CRITICAL=50)
logger = get_logger(name=__file__, console_handler_level=logging.DEBUG, file_handler_level=None)

def setup_val_loader_and_post_transforms(imagesTs_path_list, scale_min, scale_max, organ_dict):
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

    for organ in organ_dict.keys():
        channel_num = organ_dict[organ]['channel']
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
        organ_dict[organ]['post_transforms'] = post_transforms

    return val_loader, organ_dict

def setup_models(organ_dict, gpu_num=0):
    for organ in organ_dict.keys():
        device, mmar_dir, model = setup_model(organ, gpu_num)
        model_file_path = os.path.join(mmar_dir, "best_metric_model.pth")
        if os.path.isfile(model_file_path):
            model.load_state_dict(torch.load(model_file_path))
        organ_dict[organ]['device'] = device
        organ_dict[organ]['mmar_dir'] = mmar_dir
        organ_dict[organ]['model'] = model

    return organ_dict

def contours_to_json(organ, contours, png_file_name, image_shape, json_file_path):
    if len(contours) > 0:
        if os.path.isfile(json_file_path):
            with open(json_file_path, 'r') as json_file:
                json_dict = json.load(json_file)
        else:
            json_dict = OrderedDict()

        for idx, contour in enumerate(contours):
            if idx==0 and not os.path.isfile(json_file_path):
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

def run_infer(organ, organ_ele_dict, val_data, roi_size, slicesTs_list, labelsTs_data_dir):
    post_transforms = organ_ele_dict['post_transforms']
    device = organ_ele_dict['device']
    model = organ_ele_dict['model']
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))

    model.eval()
    with torch.no_grad():
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

                ret, thresh = cv2.threshold(image, 0, 255, 0)
                image_bgr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
                erosion = cv2.erode(image_bgr, kernel, iterations = 2)
                dilation = cv2.dilate(erosion, kernel, iterations = 2)

                image_gray = cv2.cvtColor(dilation, cv2.COLOR_BGR2GRAY)
                contours, heirarchy = cv2.findContours(image_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours_to_json(organ, contours, png_file_name, image.shape, json_file_path)

def infer(data_dir_list, organ_list, progress_manager_dict=None, update_progress_func=None, return_list=None):
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #   organ_dict = {                                      #
    #                   'liver': {                          #
    #                               'channel': ,            #
    #                               'post_transforms': ,    #
    #                               'device': ,             #
    #                               'mmar_dir': ,           #
    #                               'model': ,              #
    #                            },                         #
    #                   'pancreas': {},                     #
    #                   'spleen': {},                       #
    #                }                                      #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    organ_dict = OrderedDict()


    # # # # # # # # # # #
    #   Configuration   #
    # # # # # # # # # # #
    # 1: liver / 3: pancreas / 4: spleen / 5: kidney
    roi_size = config['roi_size']

    for organ in list(organ_list):
        if organ not in config['organ_to_mmar'].keys():
            organ_list.remove(organ)
        else:
            organ_dict[organ] = OrderedDict()
            organ_dict[organ]['channel'] = config['organ_to_mmar'][organ]['channel']
    logger.info("Effective organ_list is {}".format(organ_list))
    assert len(organ_list)>0, "organ_list cannot be empty"


    # # # # # # # # # # # # #
    #   Data preprocess     #
    # # # # # # # # # # # # #
    imagesTs_dir = os.path.join(test_data_dir, 'imagesTs')
    slicesTs_dir = os.path.join(test_data_dir, 'slicesTs')

    scale_min, scale_max, imagesTs_path_list, slicesTs_path_list = \
        data_preprocess_for_inference(data_dir_list, imagesTs_dir, slicesTs_dir)


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #   Setup DataLoader and post transforms for validation #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    val_loader, organ_dict = \
        setup_val_loader_and_post_transforms(imagesTs_path_list, scale_min, scale_max, organ_dict)


    # # # # # # # # # # #
    #   Create Models   #
    # # # # # # # # # # #
    organ_dict = setup_models(organ_dict, gpu_num=0)


    # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #   Check best model output with the input image    #
    # # # # # # # # # # # # # # # # # # # # # # # # # # #
    labelsTs_dir = os.path.join(test_data_dir, 'labelsTs')
    labelsTs_data_dir_list = []
    for idx, (val_data, slicesTs_path) in enumerate(zip(val_loader, slicesTs_path_list)):
        if progress_manager_dict != None and update_progress_func != None:
            progress_enable = True
            progress_dict = {}
        else:
            progress_enable = False

        if progress_enable:
            progress_dict.update(progress_manager_dict)
            if idx==0:
                progress_dict["finished"] = 0
            progress_dict["asset_group"][idx]["status"] = "processing"
            progress_dict["asset_group"][idx]["message"] = "Inference start"
            progress_manager_dict.update(progress_dict)
            update_progress_func(progress_dict)

        # Create labelsTs_data_dir
        csv_file_name = os.path.basename(slicesTs_path)
        data_dir_name, ext = os.path.splitext(csv_file_name)
        labelsTs_data_dir = os.path.join(labelsTs_dir, data_dir_name)
        labelsTs_data_dir_list.append(labelsTs_data_dir)
        if return_list != None:
            return_list.append(labelsTs_data_dir)
        os_makedirs(labelsTs_data_dir)

        # Prepare slicesTs_list to rename json file
        slicesTs_list = []
        with open(slicesTs_path, 'r') as csvfile:
            rows = csv.reader(csvfile)
            for row in rows:
                slicesTs_list.append(row)

        logger.info("Inference on {} starts".format(data_dir_name))
        for organ in organ_dict.keys():
            run_infer(organ, organ_dict[organ], val_data, roi_size, slicesTs_list, labelsTs_data_dir)

        if progress_enable:
            progress_dict["finished"] = idx + 1
            progress_dict["asset_group"][idx]["status"] = "processing"
            progress_dict["asset_group"][idx]["message"] = "Inference done"
            progress_manager_dict.update(progress_dict)
            update_progress_func(progress_dict)

        logger.info("Inference on {} is done".format(data_dir_name))

    return labelsTs_data_dir_list

if __name__ == "__main__":
    data_dir_list = ['/home/aoi/opencv_practice/monai/CT_organ_nckuh/004',
                     '/home/aoi/opencv_practice/monai/CT_organ_nckuh/003',
                     '/home/aoi/opencv_practice/monai/CT_organ_nckuh/005']

    organ_list = ['liver', 'pancreas', 'spleen']
    infer(data_dir_list, organ_list)
