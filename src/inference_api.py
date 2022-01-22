#!/usr/bin/python3
import os
import sys
import logging
import json
from multiprocessing import Process, Manager
import threading
import zipfile
import glob
from sanic import Sanic
from sanic import response

import boto3
from botocore.utils import is_valid_endpoint_url
from botocore.client import Config
from botocore.exceptions import ClientError, EndpointConnectionError

from inference import infer
# from point_reduction import get_contour_point_num, binary_search_coeff, merge_list, insert_list, sample_from_xy_list
from afs import models

current_dir = os.path.dirname(os.path.abspath(__file__))
monai_dir = os.path.dirname(current_dir)

sys.path.append(os.path.join(monai_dir, "config"))
from config import read_config_yaml, write_config_yaml, write_config_yaml_with_key_value

sys.path.append(os.path.join(monai_dir, "utils"))
from utils import os_makedirs
from logger import get_logger

# Read config_file
config_file = os.path.join(monai_dir, 'config/config.yaml')
config = read_config_yaml(config_file)
test_data_dir = config['test_data_dir']
imagesTs_dcm_dir = os.path.join(test_data_dir, 'imagesTs_dcm')
os_makedirs(imagesTs_dcm_dir, keep_exists=True)

# Get logger
# logging level (NOTSET=0 ; DEBUG=10 ; INFO=20 ; WARNING=30 ; ERROR=40 ; CRITICAL=50)
logger = get_logger(name=__file__, console_handler_level=logging.DEBUG, file_handler_level=None)


def connect_to_s3(endpoint, access_key, secret_key):
    config = Config(signature_version='s3')
    is_verify = True
    connection = boto3.client(
        's3',
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        endpoint_url=endpoint,
        config=config,
        verify=is_verify
    )
    return connection

def download_s3_file(connection, bucket, s3_file_path, docker_file_path):
    try:
        connection.download_file(bucket, s3_file_path, docker_file_path)
    except ClientError as e:
        pass

    if os.path.isfile(docker_file_path):
        return True
    else:
        return False

def upload_s3_file(connection, bucket, docker_file_path, s3_file_path):
    try:
        connection.upload_file(docker_file_path, bucket, s3_file_path)
        return True
    except ClientError as e:
        return False


class MonaiApp(object):
    def __init__(self):
        self.proc = None
        self.stop_proc = False

        self.endpoint = None
        self.access_key = None
        self.secret_key = None
        self.bucket = None
        self.organs = None
        self.asset_group = None
        self.output_s3_folder = None
        self.connection = None

        self.monai_func_thread = None
        self.event_obj = threading.Event()

        self.progress_json_path = None
        self.progress_s3_path = None

        self.app = Sanic(__name__)
        self.app.add_route(self.start_inference, "/predict", methods=['POST'])
        self.app.add_route(self.stop_inference, "/stop", methods=['POST'])
        self.app.add_route(self.download_model, "/model", methods=['POST'])
        self.app.add_route(self.get_label, "/label", methods=['POST', 'GET'])

    def run(self, host, port):
        self.app.run(host = host, port = port)

    def update_progress(self, progress_dict):
        with open(self.progress_json_path, 'w') as json_file:
            json.dump(progress_dict, json_file, sort_keys=False, indent=2, separators=(',', ': '))
        upload_s3_file(self.connection, self.bucket, self.progress_json_path, self.progress_s3_path)

    def stop_infer_process(self, event_obj):
        event_obj.wait()
        if self.proc!=None:
            self.proc.kill()

    def run_inference(self):
        self.event_obj.clear()
        stop_infer_process_thread = threading.Thread(target=self.stop_infer_process, args=(self.event_obj,))
        stop_infer_process_thread.start()

        # prepare empty progress_dict
        result_json = []
        progress_dict = {"total": "", "finished": 0, "interrupted": 0, "asset_group": ""}
        for asset in self.asset_group:
            category_name = asset.get("category_name", None)
            result_json.append({"category_name": category_name, "status": "", "message": ""})
        progress_dict["total"] = len(self.asset_group)
        progress_dict["finished"] = 0
        progress_dict["asset_group"] = result_json

        logfile_name = self.output_s3_folder.split('/')[-1]
        self.progress_json_path = os.path.join(test_data_dir, '{}_log.json'.format(logfile_name))
        self.progress_s3_path = os.path.join(self.output_s3_folder, '{}_log.json'.format(logfile_name))
        self.update_progress(progress_dict)

        if self.stop_proc:
            progress_dict["interrupted"] = 1
            self.update_progress(progress_dict)
            return

        # Download file from S3 blob
        dcm_dir_list = []
        category_name_list = []
        for idx, asset in enumerate(self.asset_group):
            category_name = asset.get("category_name", None)
            category_name_list.append(category_name)
            files = asset.get("files", None)

            dcm_dir_name = files[0].split('/')[-2]
            dcm_dir = os.path.join(imagesTs_dcm_dir, dcm_dir_name)
            dcm_dir_list.append(dcm_dir)
            os_makedirs(dcm_dir)

            logger.info("Downloading {} folder from s3 blob to {}".format(dcm_dir_name, dcm_dir))
            if idx==0:
                progress_dict["finished"] = 0
            result_json[idx]["status"] = "processing"
            result_json[idx]["message"] = "Downloading {} folder from s3 blob to {}".format(dcm_dir_name, dcm_dir)
            progress_dict["asset_group"] = result_json
            self.update_progress(progress_dict)

            for s3_file_path in files:
                if self.stop_proc:
                    result_json[idx]["status"] = "fail"
                    result_json[idx]["message"] = "Stop by user"
                    progress_dict["asset_group"] = result_json
                    progress_dict["interrupted"] = 1
                    self.update_progress(progress_dict)
                    return

                dcm_file_name = os.path.basename(s3_file_path)
                dcm_file_path = os.path.join(dcm_dir, dcm_file_name)
                if not download_s3_file(self.connection, self.bucket, s3_file_path, dcm_file_path):
                    result_json[idx]["status"] = "fail"
                    result_json[idx]["message"] = "Cannot download {} from s3 blob".format(s3_file_path)
                    progress_dict["asset_group"] = result_json
                    progress_dict["interrupted"] = 1
                    self.update_progress(progress_dict)
                    return

            result_json[idx]["message"] = "Downloading from s3 blob is done"
            progress_dict["asset_group"] = result_json
            progress_dict["finished"] = idx + 1
            self.update_progress(progress_dict)

        # Inference
        # labelsTs_data_dir_list = infer(dcm_dir_list, self.organs, progress_dict, self.update_progress)
        logger.info("Inference procedure starts")
        manager = Manager()
        progress_manager_dict = manager.dict()
        progress_manager_dict.update(progress_dict)
        labelsTs_data_dir_list = manager.list()
        self.proc = Process(target=infer, args=(dcm_dir_list, self.organs, progress_manager_dict, self.update_progress, labelsTs_data_dir_list))
        self.proc.start()
        self.proc.join()

        progress_dict.update(progress_manager_dict)
        if self.proc.exitcode!=0:
            logger.info("Inference is interrupted")
            for idx in range(len(result_json)):
                result_json[idx]["status"] = "fail"
                result_json[idx]["message"] = "Stop by user"
            progress_dict["asset_group"] = result_json
            progress_dict["interrupted"] = 1
            self.update_progress(progress_dict)
            return
        logger.info("All inferences are done")


        # Upload file to S3 blob
        done_count = 0
        for idx, (category_name, labelsTs_data_dir) in enumerate(zip(category_name_list, labelsTs_data_dir_list)):
            logger.info("Uploading {} folder to s3 blob".format(labelsTs_data_dir))
            if idx==0:
                progress_dict["finished"] = 0
            result_json[idx]["status"] = "processing"
            result_json[idx]["message"] = "Uploading {} folder to s3 blob".format(labelsTs_data_dir)
            progress_dict["asset_group"] = result_json
            self.update_progress(progress_dict)

            s3_file_dir = os.path.join(self.output_s3_folder, category_name, "json")
            finish = True
            for json_file in os.listdir(labelsTs_data_dir):
                if self.stop_proc:
                    result_json[idx]["status"] = "fail"
                    result_json[idx]["message"] = "Stop by user"
                    progress_dict["asset_group"] = result_json
                    progress_dict["interrupted"] = 1
                    self.update_progress(progress_dict)
                    return
                json_file_path = os.path.join(labelsTs_data_dir, json_file)
                s3_file_path = os.path.join(s3_file_dir, json_file)
                if not upload_s3_file(self.connection, self.bucket, json_file_path, s3_file_path):
                    result_json[idx]["status"] = "fail"
                    result_json[idx]["message"] = "Cannot upload {} to s3 blob".format(json_file_path)
                    progress_dict["asset_group"] = result_json
                    progress_dict["interrupted"] = 1
                    self.update_progress(progress_dict)
                    finish = False

            if finish:
                done_count = done_count + 1
                result_json[idx]["status"] = "finished"
                result_json[idx]["message"] = "Uploading to S3 blob is done"
                progress_dict["asset_group"] = result_json
                progress_dict["finished"] = done_count
                self.update_progress(progress_dict)
        logger.info("All done")
        self.event_obj.set()

    # @app.route("/predict", methods=['POST'])
    async def start_inference(self, request):
        supported_organ_list = list(config['organ_to_mmar'].keys())
        empty_response = {"message": ""}

        request_dict = request.json
        self.organs = request_dict.get('organs', supported_organ_list)
        self.endpoint = request_dict.get('endpoint', None)
        self.access_key = request_dict.get('access_key', None)
        self.secret_key = request_dict.get('secret_key', None)
        self.bucket = request_dict.get('bucket', None)
        self.asset_group = request_dict.get('asset_group', None)
        self.output_s3_folder = request_dict.get('output_s3_folder', None)

        # Check if the input organs to be inferred are all supported
        unsupported_organ_list = []
        for organ in list(self.organs):
            if organ not in supported_organ_list:
                unsupported_organ_list.append(organ)
                self.organs.remove(organ)

        if len(unsupported_organ_list) > 0:
            empty_response["message"] = "Fail: Does not support '{}' inference. Support only {}"\
                                        .format(', '.join(unsupported_organ_list), ', '.join(supported_organ_list))
            return response.json(empty_response)

        config['organ_list'] = self.organs
        write_config_yaml_with_key_value(config_file, 'organ_list', self.organs)

        # Check if the S3 credentials
        if self.endpoint and self.access_key and self.secret_key and self.bucket and self.asset_group:
            if isinstance(self.asset_group, list):
                self.connection = connect_to_s3(self.endpoint, self.access_key, self.secret_key)
                self.monai_func_thread = threading.Thread(target=self.run_inference)
                self.monai_func_thread.start()
                empty_response["message"] = "Success: Total {} patients will be dealt with by batch mode".format(len(self.asset_group))
                return response.json(empty_response)
            else:
                empty_response["message"] = "Fail: asset_group is not list"
                return response.json(empty_response)
        else:
            msg_list = []
            if not self.endpoint:
                msg_list.append("endpoint")
            if not self.access_key:
                msg_list.append("access_key")
            if not self.secret_key:
                msg_list.append("secret_key")
            if not self.bucket:
                msg_list.append("bucket")
            if not self.asset_group:
                msg_list.append("asset_group")

            empty_response["message"] = "Fail: Lack of {}".format(', '.join(msg_list))
            return response.json(empty_response)

        empty_response["message"] = "Success: inference is ongoing"
        return response.json(empty_response)

    # @app.route("/stop", methods=['POST'])
    async def stop_inference(self, request):
        self.stop_proc = True
        self.event_obj.set()
        if self.monai_func_thread != None:
            self.monai_func_thread.join()
        self.stop_proc = False
        return response.json({"message": "Success: inference is interrupted"})

    # @app.route("/model", methods=['POST'])
    async def download_model(self, request):
        models_dir = config['models_dir']
        request_dict = request.json
        logger.info("Download model request: {}".format(request_dict))

        save_dir = request_dict['save_model_to']
        model_repository_name = request_dict['name']
        model_name = request_dict['version']
        update_model_zip_path = os.path.join(models_dir, model_repository_name)

        afs_models = models()
        afs_models.download_model(save_path=update_model_zip_path, model_repository_name=model_repository_name, model_name=model_name)

        if os.path.isfile(update_model_zip_path):
            update_model_dir = os.path.join(models_dir, 'inference')
            os_makedirs(update_model_dir)
            with zipfile.ZipFile(update_model_zip_path) as zf:
                zf.extractall(update_model_dir)

            for organ in config['organ_to_mmar']:
                new_model_dir = config['organ_to_mmar'][organ]['new_model_dir']
                if new_model_dir != None:
                    new_model_file_path_list = glob.glob(os.path.join(new_model_dir, "*.pth"))
                    if len(new_model_file_path_list) > 0:
                        config['organ_to_mmar'][organ]['new_model_file_path'] = new_model_file_path_list[0]
            write_config_yaml(config_file, config)
        return response.json({'status': 'success'})

    # @app.route("/label", methods=['POST', 'GET'])
    async def get_label(self, request):
        return response.text("monai_segmentation")

if __name__ == "__main__":
    app = MonaiApp()
    port = int(os.environ.get('cloud_inference_port', 1234))
    app.run(host = '0.0.0.0', port = port)
