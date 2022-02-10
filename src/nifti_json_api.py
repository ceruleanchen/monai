#!/usr/bin/python3
import os
import sys
import logging
import json
import threading
import zipfile
import glob
import pydicom
import dicom2nifti
import nibabel as nib
import csv
import cv2
import numpy as np
from sanic import Sanic
from sanic import response
import base64

from inference import contours_to_json, empty_json
from afs2datasource import DBManager, constant

import nest_asyncio
nest_asyncio.apply()

current_dir = os.path.dirname(os.path.abspath(__file__))
monai_dir = os.path.dirname(current_dir)

sys.path.append(os.path.join(monai_dir, "config"))
from config import read_config_yaml, write_config_yaml, write_config_yaml_with_key_value

sys.path.append(os.path.join(monai_dir, "utils"))
from utils import os_makedirs, os_remove, shutil_copyfile, shutil_rmtree
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
            nifti_file_path_list = glob.glob('{}/*.nii.gz'.format(data_dir))
            for nifti_idx, nifti_file_path in enumerate(nifti_file_path_list):
                nifti_file_name = os.path.basename(nifti_file_path) # '001_liver.nii.gz'
                nifti_file_id = nifti_file_name.split('.')[0]       # '001_liver'
                organ = nifti_file_id.split('_')[1]                 # 'liver'

                image_arr = nib.load(nifti_file_path).get_data()
                num_slices = image_arr.shape[2]
                for slice_idx in range(num_slices):
                    image = np.flip(image_arr[:,:,slice_idx].astype('uint8').T, axis=0)
                    slice_location, dcm_file_name = slices_list[slice_idx]
                    json_file_name = dcm_file_name.replace('.dcm', '.json')
                    json_file_path = os.path.join(data_dir, json_file_name)
                    png_file_name = dcm_file_name.replace('.dcm', '.png')
                    if nifti_idx==0:
                        os_remove(json_file_path)

                    if np.any(image>0):
                        ret, thresh = cv2.threshold(image, 0, 255, 0)
                        image_bgr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
                        erosion = cv2.erode(image_bgr, kernel, iterations = 2)
                        dilation = cv2.dilate(erosion, kernel, iterations = 2)

                        image_gray = cv2.cvtColor(dilation, cv2.COLOR_BGR2GRAY)
                        contours, heirarchy = cv2.findContours(image_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        contours_to_json(organ, contours, png_file_name, image.shape, json_file_path)
                    elif nifti_idx==0:
                        empty_json(png_file_name, image.shape, json_file_path)

                    if nifti_idx==len(nifti_file_path_list)-1:
                        s3_json_file_path = os.path.join(s3_folder, json_file_name)
                        print(slice_idx, s3_json_file_path)
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
                empty_response["message"] = "Success: Total {} patients will be imported by batch mode".format(len(folder))
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

    def run_export(self, endpoint, access_key, secret_key, bucket, asset_group):
        empty_response = {"asset_group": []}

        for asset in asset_group:
            category_name = asset.get("category_name", None)
            files = asset.get("files", None)
            return_asset = {"category_name": category_name, "zipfile": None}

            # Download files from S3 blob
            bucket_dir = download_s3(endpoint, access_key, secret_key, bucket, file_list=files, folder_list=[])

            # Copy files from bucket_dir to export_dir
            export_dir = os.path.join(bucket_dir, 'export')
            export_category_dir = os.path.join(export_dir, category_name)
            os_makedirs(export_category_dir)

            dcm_file_list = []
            json_file_list = []
            for s3_file_path in files:
                file_name = os.path.basename(s3_file_path)
                src_file_path = os.path.join(bucket_dir, s3_file_path)
                dst_file_path = os.path.join(export_category_dir, file_name)
                if os.path.splitext(file_name)[1] == '.dcm':
                    dcm_file_list.append(dst_file_path)
                if os.path.splitext(file_name)[1] == '.json':
                    json_file_list.append(dst_file_path)
                shutil_copyfile(src_file_path, dst_file_path)

            # Convert polygon to mask
            organ_list = list(config['organ_to_mmar'].keys()) # organ_list = ['liver', 'pancreas', 'spleen']
            organ_dict = dict.fromkeys(organ_list)
            for organ in organ_dict:
                label_dir = export_category_dir + '_' + organ
                organ_dict[organ] = {'used': False, 'label_dir': label_dir}
                os_makedirs(label_dir)

            for dcm_file_path, json_file_path in zip(dcm_file_list, json_file_list):
                dcm_file_name = os.path.basename(dcm_file_path)
                ds = pydicom.read_file(dcm_file_path)
                ds.RescaleIntercept = 0
                ds.WindowCenter = 3.0
                ds.WindowWidth = 6.0
                img = ds.pixel_array
                mask_bgr = np.zeros([*img.shape, 3], dtype='uint8')
                for organ in organ_list:
                    organ_dict[organ]['mask_bgr'] = np.copy(mask_bgr)

                with open(json_file_path) as jsfile:
                    js = json.load(jsfile)

                for shape in js['shapes']:
                    label = shape['label']
                    if label not in organ_list:
                        logger.warning("label = {} does not support. Support only {}".format(label, organ_list))
                        continue

                    if organ_dict[label]['used'] == False:
                        organ_dict[label]['used'] = True

                    contour = np.expand_dims(np.array(shape['points']), axis=1)
                    contour = contour.astype('int32')
                    val = 1
                    cv2.drawContours(organ_dict[label]['mask_bgr'], [contour], -1, (val,val,val),-1)

                for organ in organ_list:
                    mask_gray = cv2.cvtColor(organ_dict[organ]['mask_bgr'], cv2.COLOR_BGR2GRAY)
                    mask_gray = mask_gray.astype('int16') # should be numpy.float64?
                    ds.PixelData = mask_gray
                    ds.save_as(os.path.join(organ_dict[organ]['label_dir'], dcm_file_name))

            # Convert mask to nifti
            nifti_file_path_list = []
            for organ in organ_dict:
                label_dir = organ_dict[organ]['label_dir']
                if organ_dict[organ]['used']:
                    label_dir_name = os.path.basename(label_dir)
                    nifti_file_path = os.path.join(label_dir, '{}.nii.gz'.format(label_dir_name))
                    nifti_file_path_list.append(nifti_file_path)
                    dicom2nifti.dicom_series_to_nifti(label_dir, nifti_file_path, reorient_nifti=True)
                else:
                    shutil_rmtree(label_dir)

            # Zip nifti_file_path_list
            zip_file_path = os.path.join(export_category_dir, '{}.zip'.format(category_name))
            with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                for nifti_file_path in nifti_file_path_list:
                    zf.write(nifti_file_path, os.path.basename(nifti_file_path))

            # Serialize Zipfile
            with open(zip_file_path, "rb") as zf:
                zf_bytes = zf.read()
                zf_b64 = base64.b64encode(zf_bytes).decode("utf8")
                return_asset["zipfile"] = zf_b64
            empty_response["asset_group"].append(return_asset)

        return empty_response

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
                '''
                self.export_func_thread = threading.Thread(target=self.run_export, args=(endpoint, access_key, secret_key, bucket, asset_group))
                self.export_func_thread.start()
                self.export_func_thread.join()
                empty_response["message"] = "Success: Total {} patients have been exported by batch mode".format(len(asset_group))
                return response.json(empty_response)
                '''
                empty_response = self.run_export(endpoint, access_key, secret_key, bucket, asset_group)
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


if __name__ == "__main__":
    app = NiftiApp()
    port = 1234
    app.run(host = '0.0.0.0', port = port)
