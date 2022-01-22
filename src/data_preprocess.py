import os, sys
import glob
import json
import numpy as np
import csv
import cv2
import logging
import dicom2nifti
import pydicom
import nibabel as nib
from typing import Callable, Dict, List, Optional, Sequence, Union
# https://myapollo.com.tw/zh-tw/python-typing-module/

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
logger = get_logger(name=__file__, console_handler_level=logging.INFO, file_handler_level=None)

def data_preprocess_for_training(dataset_dir, imagesTr_dir, labelsTr_dir, slicesTr_dir, scale_min=float('inf'), scale_max=float('-inf')):
    # makedirs
    os_makedirs(imagesTr_dir)
    os_makedirs(labelsTr_dir)
    os_makedirs(slicesTr_dir)

    organ_list = ['all']
    organ_list.extend(list(config['organ_to_mmar'].keys())) # organ_list = ['all', 'liver', 'pancreas', 'spleen']
    organ_dict = dict.fromkeys(organ_list)
    for organ in organ_dict:
        organ_dict[organ] = {}

    # Assume only the 001 ~ 005 data have this kind of label
    # 1: liver / 3: pancreas / 4: spleen / 5: kidney
    # obsolete_label_to_organ_mapping = {'1': 'liver', '3': 'pancreas', '4': 'spleen', '5': 'kidney', 'd': 'kidney'}
    obsolete_label_to_organ_mapping = {'1': 'liver', '3': 'pancreas', '4': 'spleen'}

    for data_dir_name in os.listdir(dataset_dir):
        data_dir = os.path.join(dataset_dir, data_dir_name)

        if os.path.isdir(data_dir):
            # imagesTr
            imagesTr_path = os.path.join(imagesTr_dir, '{}.nii.gz'.format(data_dir_name))
            dicom2nifti.dicom_series_to_nifti(data_dir, imagesTr_path, reorient_nifti=True)

            # labelsTr
            for organ in organ_list:
                labelsTr_organ_dir = os.path.join(labelsTr_dir, '{}_{}'.format(data_dir_name, organ))
                os_makedirs(labelsTr_organ_dir)
                organ_dict[organ]['label_dir'] = labelsTr_organ_dir

            dcm_file_list = sorted(glob.glob('{}/*.dcm'.format(data_dir)))
            json_file_list = sorted(glob.glob('{}/*.json'.format(data_dir)))

            slice_location_to_file_name_mapping = dict()

            for dcm_file_path in dcm_file_list:
                dcm_file_name = os.path.basename(dcm_file_path)
                ds = pydicom.read_file(dcm_file_path)

                scale_min = min(int(ds.WindowCenter) - int(ds.WindowWidth)//2, scale_min)
                scale_max = max(int(ds.WindowCenter) + int(ds.WindowWidth)//2, scale_max)

                ds.RescaleIntercept = 0
                ds.WindowCenter = 3.0
                ds.WindowWidth = 6.0
                slice_location_to_file_name_mapping[float(ds.SliceLocation)] = dcm_file_name
                logger.debug("{:>20s}, {:>3s}, {:>10s}".format(dcm_file_name, str(ds.InstanceNumber), str(ds.SliceLocation)))

                # Convert polygon to mask
                img = ds.pixel_array
                mask_bgr = np.zeros([*img.shape, 3], dtype='uint8')
                for organ in organ_list:
                    organ_dict[organ]['mask_bgr'] = np.copy(mask_bgr)
                json_file_path = dcm_file_path.replace('.dcm', '.json')

                if json_file_path in json_file_list:
                    with open(json_file_path) as jsfile:
                        js = json.load(jsfile)

                    for shape in js['shapes']:
                        label = shape['label']
                        organ = obsolete_label_to_organ_mapping.get(label, label)

                        if organ not in organ_list:
                            logger.warning("label = {} does not support. Support only {}".format(organ, organ_list[1:]))
                            continue

                        contour = np.expand_dims(np.array(shape['points']), axis=1)
                        contour = contour.astype('int32')
                        val = 1
                        cv2.drawContours(organ_dict[organ]['mask_bgr'], [contour], -1, (val,val,val),-1)
                        val = organ_list.index(organ)
                        cv2.drawContours(organ_dict['all']['mask_bgr'], [contour], -1, (val,val,val),-1)

                for organ in organ_list:
                    mask_gray = cv2.cvtColor(organ_dict[organ]['mask_bgr'], cv2.COLOR_BGR2GRAY)
                    mask_gray = mask_gray.astype('int16') # should be numpy.float64?
                    ds.PixelData = mask_gray
                    ds.save_as(os.path.join(organ_dict[organ]['label_dir'], dcm_file_name))

            for organ in organ_list:
                labelsTr_path = organ_dict[organ]['label_dir'] + '.nii.gz'
                dicom2nifti.dicom_series_to_nifti(organ_dict[organ]['label_dir'], labelsTr_path, reorient_nifti=True)

            # slicesTr
            slicesTr_path = os.path.join(slicesTr_dir, '{}.csv'.format(data_dir_name))
            with open(slicesTr_path, 'w') as csvfile:
                writer = csv.writer(csvfile)
                for slice_location in sorted(slice_location_to_file_name_mapping.keys()):
                    writer.writerow([slice_location, slice_location_to_file_name_mapping[slice_location]])

    config['scale_min'] = scale_min
    config['scale_max'] = scale_max
    write_config_yaml(config_file, config)
    return scale_min, scale_max

def data_preprocess_for_inference(data_dir_list, imagesTs_dir, slicesTs_dir):
    # makedirs
    os_makedirs(imagesTs_dir, keep_exists=True)
    os_makedirs(slicesTs_dir, keep_exists=True)

    imagesTs_path_list = []
    slicesTs_path_list = []
    scale_min = float('inf')
    scale_max = float('-inf')

    for data_dir in data_dir_list:
        # imagesTs
        data_dir_name = os.path.basename(data_dir)
        imagesTs_path = os.path.join(imagesTs_dir, '{}.nii.gz'.format(data_dir_name))
        imagesTs_path_list.append(imagesTs_path)
        dicom2nifti.dicom_series_to_nifti(data_dir, imagesTs_path, reorient_nifti=True)

        # slicesTs
        dcm_file_list = sorted(glob.glob('{}/*.dcm'.format(data_dir)))
        slice_location_to_file_name_mapping = dict()
        for dcm_file_path in dcm_file_list:
            dcm_file_name = os.path.basename(dcm_file_path)
            ds = pydicom.read_file(dcm_file_path)

            scale_min = min(int(ds.WindowCenter) - int(ds.WindowWidth)//2, scale_min)
            scale_max = max(int(ds.WindowCenter) + int(ds.WindowWidth)//2, scale_max)
            slice_location_to_file_name_mapping[float(ds.SliceLocation)] = dcm_file_name

        slicesTs_path = os.path.join(slicesTs_dir, '{}.csv'.format(data_dir_name))
        slicesTs_path_list.append(slicesTs_path)
        with open(slicesTs_path, 'w') as csvfile:
            writer = csv.writer(csvfile)
            for slice_location in sorted(slice_location_to_file_name_mapping.keys()):
                writer.writerow([slice_location, slice_location_to_file_name_mapping[slice_location]])

    return scale_min, scale_max, imagesTs_path_list, slicesTs_path_list

if __name__ == '__main__':
    # Prepare train_data
    dataset_dir = os.path.join(config['monai_dataset_dir'], 'train_data')
    imagesTr_dir = os.path.join(config['train_data_dir'], 'imagesTr')
    labelsTr_dir = os.path.join(config['train_data_dir'], 'labelsTr')
    slicesTr_dir = os.path.join(config['train_data_dir'], 'slicesTr')
    scale_min, scale_max = data_preprocess_for_training(dataset_dir, imagesTr_dir, labelsTr_dir, slicesTr_dir)

    # Prepare val_data
    dataset_dir = os.path.join(config['monai_dataset_dir'], 'val_data')
    imagesTr_dir = os.path.join(config['val_data_dir'], 'imagesTr')
    labelsTr_dir = os.path.join(config['val_data_dir'], 'labelsTr')
    slicesTr_dir = os.path.join(config['val_data_dir'], 'slicesTr')
    data_preprocess_for_training(dataset_dir, imagesTr_dir, labelsTr_dir, slicesTr_dir, scale_min, scale_max)
