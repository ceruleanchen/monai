#!/usr/bin/python3
import os
import sys
import logging
import json
from multiprocessing import Process, Manager
import threading
import zipfile
import glob
import pydicom
import nibabel as nib
import csv
import cv2
import numpy as np
from sanic import Sanic
from sanic import response

import boto3
from botocore.utils import is_valid_endpoint_url
from botocore.client import Config
from botocore.exceptions import ClientError, EndpointConnectionError

from inference import contours_to_json, empty_json
from afs2datasource import DBManager, constant

import nest_asyncio
nest_asyncio.apply()

current_dir = os.path.dirname(os.path.abspath(__file__))
monai_dir = os.path.dirname(current_dir)

sys.path.append(os.path.join(monai_dir, "config"))
from config import read_config_yaml, write_config_yaml, write_config_yaml_with_key_value

sys.path.append(os.path.join(monai_dir, "utils"))
from utils import os_makedirs, os_remove, shutil_copyfile
from logger import get_logger

# Read config_file
config_file = os.path.join(monai_dir, 'config/config.yaml')
config = read_config_yaml(config_file)

# Get logger
# logging level (NOTSET=0 ; DEBUG=10 ; INFO=20 ; WARNING=30 ; ERROR=40 ; CRITICAL=50)
logger = get_logger(name=__file__, console_handler_level=logging.DEBUG, file_handler_level=None)


def download_s3(endpoint, access_key, secret_key, bucket, file_list=[], folder_list=[]):
    db = DBManager(
        db_type=constant.DB_TYPE['S3'],
        endpoint=endpoint,
        access_key=access_key,
        secret_key=secret_key,
        buckets=[{
            'bucket': bucket,
            'blobs': {
                'files': file_list,
                'folders': folder_list
            }
        }]
    )
    db.connect()
    bucket = db.execute_query()[0]
    return os.path.join(current_dir, bucket)

def upload_s3(endpoint, access_key, secret_key, bucket, src_file_path, dst_file_path):
    db = DBManager(
        db_type=constant.DB_TYPE['S3'],
        endpoint=endpoint,
        access_key=access_key,
        secret_key=secret_key,
        buckets=[{
            'bucket': bucket,
            'blobs': {
                'files': [],
                'folders': []
            }
        }]
    )
    db.connect()
    db.insert(table_name=bucket, source=src_file_path, destination=dst_file_path)

class NiftiApp(object):
    def __init__(self):
        self.proc = None
        self.stop_proc = False

        self.import_func_thread = None
        self.export_func_thread = None
        self.event_obj = threading.Event()

        self.app = Sanic(__name__)
        # Import (nifti to json)
        self.app.add_route(self.start_import, "/import", methods=['POST'])
        # Export (json to nifti)
        self.app.add_route(self.start_export, "/export", methods=['POST'])

    def run(self, host, port):
        self.app.run(host = host, port = port)

    # Import (nifti to json)
    def run_import(self, endpoint, access_key, secret_key, bucket, folder_list):
        # Download file from S3 blob
        bucket_dir = download_s3(endpoint, access_key, secret_key, bucket, file_list=[], folder_list=folder_list)

        # Iterate through s3 folders
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
        for s3_folder in folder_list:
            data_dir = os.path.join(bucket_dir, s3_folder)
            data_dir_name = os.path.basename(data_dir)
            dcm_file_list = sorted(glob.glob('{}/*.dcm'.format(data_dir)))

            # slice location to file name mapping
            slice_location_to_file_name_mapping = dict()
            for dcm_file_path in dcm_file_list:
                dcm_file_name = os.path.basename(dcm_file_path)
                ds = pydicom.read_file(dcm_file_path)
                slice_location_to_file_name_mapping[float(ds.SliceLocation)] = dcm_file_name

            slices_csv_path = os.path.join(data_dir, '{}.csv'.format(data_dir_name))
            slices_list = list()
            with open(slices_csv_path, 'w') as csvfile:
                writer = csv.writer(csvfile)
                for slice_location in sorted(slice_location_to_file_name_mapping.keys()):
                    writer.writerow([slice_location, slice_location_to_file_name_mapping[slice_location]])
                    slices_list.append([slice_location, slice_location_to_file_name_mapping[slice_location]])

            # nifti to json
            nifti_file_path = glob.glob('{}/*.nii.gz'.format(data_dir))[0]
            image_arr = nib.load(nifti_file_path).get_data()
            num_slices = image_arr.shape[2]
            for idx in range(num_slices):
                image = np.flip(image_arr[:,:,idx].astype('uint8').T, axis=0)
                slice_location, dcm_file_name = slices_list[idx]
                json_file_name = dcm_file_name.replace('.dcm', '.json')
                json_file_path = os.path.join(data_dir, json_file_name)
                png_file_name = dcm_file_name.replace('.dcm', '.png')
                os_remove(json_file_path)

                if np.any(image>0):
                    ret, thresh = cv2.threshold(image, 0, 255, 0)
                    image_bgr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
                    erosion = cv2.erode(image_bgr, kernel, iterations = 2)
                    dilation = cv2.dilate(erosion, kernel, iterations = 2)

                    image_gray = cv2.cvtColor(dilation, cv2.COLOR_BGR2GRAY)
                    contours, heirarchy = cv2.findContours(image_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    contours_to_json('1', contours, png_file_name, image.shape, json_file_path)
                else:
                    empty_json(png_file_name, image.shape, json_file_path)

                s3_json_file_path = os.path.join(s3_folder, json_file_name)
                upload_s3(endpoint, access_key, secret_key, bucket, json_file_path, s3_json_file_path)

    # Import (nifti to json)
    # @app.route("/import", methods=['POST'])
    async def start_import(self, request):
        empty_response = {"message": ""}

        request_dict = request.json
        endpoint = request_dict.get('endpoint', None)
        access_key = request_dict.get('access_key', None)
        secret_key = request_dict.get('secret_key', None)
        bucket = request_dict.get('bucket', None)
        folder = request_dict.get('folder', None)

        # Check if the S3 credentials
        if endpoint and access_key and secret_key and bucket:
            if isinstance(folder, list):
                self.import_func_thread = threading.Thread(target=self.run_import, args=(endpoint, access_key, secret_key, bucket, folder))
                self.import_func_thread.start()
                empty_response["message"] = "Success: Total {} patients will be dealt with by batch mode".format(len(folder))
                return response.json(empty_response)
            else:
                empty_response["message"] = "Fail: folder is not list"
                return response.json(empty_response)
        else:
            msg_list = []
            if not endpoint:
                msg_list.append("endpoint")
            if not access_key:
                msg_list.append("access_key")
            if not secret_key:
                msg_list.append("secret_key")
            if not bucket:
                msg_list.append("bucket")

            empty_response["message"] = "Fail: Lack of {}".format(', '.join(msg_list))
            return response.json(empty_response)

        empty_response["message"] = "Success: inference is ongoing"
        return response.json(empty_response)

    def run_export(self, endpoint, access_key, secret_key, bucket, asset_group):
        for asset in asset_group:
            category_name = asset.get("category_name", None)
            files = asset.get("files", None)

            # Download files from S3 blob
            bucket_dir = download_s3(endpoint, access_key, secret_key, bucket, file_list=files, folder_list=[])

            # Copy files from bucket_dir to export_dir
            export_dir = os.path.join(bucket_dir, 'export', category_name)
            os_makedirs(export_dir)

            dcm_file_list = []
            json_file_list = []
            for s3_file_path in files:
                file_name = os.path.basename(s3_file_path)
                src_file_path = os.path.join(bucket_dir, s3_file_path)
                dst_file_path = os.path.join(export_dir, file_name)
                if os.path.splitext(file_name)[1] == '.dcm':
                    dcm_file_list.append(dst_file_path)
                if os.path.splitext(file_name)[1] == '.json':
                    json_file_list.append(dst_file_path)
                shutil_copyfile(src_file_path, dst_file_path)


    # Export (json to nifti)
    # @app.route("/export", methods=['POST'])
    async def start_export(self, request):
        empty_response = {"message": ""}

        request_dict = request.json
        endpoint = request_dict.get('endpoint', None)
        access_key = request_dict.get('access_key', None)
        secret_key = request_dict.get('secret_key', None)
        bucket = request_dict.get('bucket', None)
        asset_group = request_dict.get('asset_group', None)

        # Check if the S3 credentials
        if endpoint and access_key and secret_key and bucket:
            if isinstance(asset_group, list):
                self.export_func_thread = threading.Thread(target=self.run_export, args=(endpoint, access_key, secret_key, bucket, asset_group))
                self.export_func_thread.start()
                self.export_func_thread.join()
                empty_response["message"] = "Success: Total {} patients have been dealt with by batch mode".format(len(asset_group))
                return response.json(empty_response)
            else:
                empty_response["message"] = "Fail: asset_group is not list"
                return response.json(empty_response)
        else:
            msg_list = []
            if not endpoint:
                msg_list.append("endpoint")
            if not access_key:
                msg_list.append("access_key")
            if not secret_key:
                msg_list.append("secret_key")
            if not bucket:
                msg_list.append("bucket")
            if not asset_group:
                msg_list.append("asset_group")

            empty_response["message"] = "Fail: Lack of {}".format(', '.join(msg_list))
            return response.json(empty_response)

        return response.json({'status': 'success'})


if __name__ == "__main__":
    app = NiftiApp()
    port = 1234
    app.run(host = '0.0.0.0', port = port)
