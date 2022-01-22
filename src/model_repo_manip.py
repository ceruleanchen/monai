#!/usr/bin/python3
import os, sys
import logging
import zipfile
from afs import models

current_dir = os.path.dirname(os.path.abspath(__file__))
monai_dir = os.path.dirname(current_dir)

sys.path.append(os.path.join(monai_dir, "config"))
from config import read_config_yaml, write_config_yaml_with_key_value

sys.path.append(os.path.join(monai_dir, "utils"))
from utils import os_makedirs, shutil_rmtree
from logger import get_logger

# Read config_file
config_file = os.path.join(monai_dir, 'config/config.yaml')
config = read_config_yaml(config_file)

# Get logger
# logging level (NOTSET=0 ; DEBUG=10 ; INFO=20 ; WARNING=30 ; ERROR=40 ; CRITICAL=50)
logger = get_logger(name=__file__, console_handler_level=logging.DEBUG, file_handler_level=None)

default_repo_name = 'monai_retrain.zip'

# def upload_to_model_repo(repo_name='pcb_retrain'):


def take_floating_ymd(elem):
    days_list = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    file_id, ext = os.path.splitext(elem)
    year, month, day = [int(num) for num in file_id.split('_')]
    days = sum(days_list[0:month]) + day
    key = year + days / sum(days_list)
    return key

def get_latest_model_from_monai_dataset(monai_dataset_model_dir):
    if not os.path.isdir(monai_dataset_model_dir):
        return None

    zip_file_list = [zipfile for zipfile in os.listdir(monai_dataset_model_dir) if zipfile.endswith('.zip')]
    zip_file_list.sort(key=take_floating_ymd, reverse=True)
    if len(zip_file_list) > 0:
        zip_file_name = zip_file_list[0]
        zip_file_path = os.path.join(monai_dataset_model_dir, zip_file_name)
        return zip_file_path
    return None

def download_from_model_repo(repo_name=default_repo_name):
    old_model_dir = os.path.join(monai_dir, 'models/old')

    monai_dataset_dir = config['monai_dataset_dir']
    monai_dataset_model_dir = os.path.join(monai_dataset_dir, 'models') 

    if config['production']=='retrain_aifs':
        try:
            afs_models = models()
            info = afs_models.get_latest_model_info(model_repository_name=repo_name)
            zip_file_path = old_model_dir + '.zip'
            afs_models.download_model(
                save_path = zip_file_path, 
                model_repository_name = repo_name, 
                last_one = True)
        except Exception as e:
            logger.warning(e)
            zip_file_path = get_latest_model_from_monai_dataset(monai_dataset_model_dir)
    else:
        zip_file_path = get_latest_model_from_monai_dataset(monai_dataset_model_dir)

    if zip_file_path==None:
        shutil_rmtree(old_model_dir)
        write_config_yaml_with_key_value(config_file, 'old_model_dir', None)
    else:
        os_makedirs(old_model_dir)
        with zipfile.ZipFile(zip_file_path, 'r') as zf:
            zf.extractall(old_model_dir)
        write_config_yaml_with_key_value(config_file, 'old_model_dir', old_model_dir)

if __name__ == "__main__":
    download_from_model_repo()