#!/usr/bin/python3
import os, sys
import logging
import zipfile
from afs import models
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
monai_dir = os.path.dirname(current_dir)

sys.path.append(os.path.join(monai_dir, "config"))
from config import read_config_yaml, write_config_yaml_with_key_value

sys.path.append(os.path.join(monai_dir, "utils"))
from utils import os_makedirs, shutil_rmtree, shutil_copytree
from logger import get_logger

# Read config_file
config_file = os.path.join(monai_dir, 'config/config.yaml')
config = read_config_yaml(config_file)

# Get logger
# logging level (NOTSET=0 ; DEBUG=10 ; INFO=20 ; WARNING=30 ; ERROR=40 ; CRITICAL=50)
logger = get_logger(name=__file__, console_handler_level=logging.DEBUG, file_handler_level=None)

default_repo_name = 'monai_retrain.zip'

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

def upload_to_model_repo(repo_name=default_repo_name):
    # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #       Copy best models to models/new folder       #
    # # # # # # # # # # # # # # # # # # # # # # # # # # #
    models_dir = config['models_dir']
    new_model_dir = os.path.join(models_dir, 'new')
    os_makedirs(new_model_dir)

    tags = {}
    supported_organ_list = list(config['organ_to_mmar'].keys())
    for organ in supported_organ_list:
        if organ in config['organ_list']:
            old_model_dice = config['organ_to_mmar'][organ]['old_model_dice']
            new_model_dice = config['organ_to_mmar'][organ]['new_model_dice']
            if new_model_dice >= old_model_dice:
                src_dir = config['organ_to_mmar'][organ]['new_model_dir']
                dst_dir = os.path.join(new_model_dir, organ)
                shutil_copytree(src_dir, dst_dir)
                tags['old/new ({})'.format(organ)] = '{}/{}'.format(old_model_dice, new_model_dice)
            else:
                old_model_file_path = config['organ_to_mmar'][organ]['old_model_file_path']
                if old_model_file_path!=None:
                    src_dir = config['organ_to_mmar'][organ]['old_model_dir']
                    dst_dir = os.path.join(new_model_dir, organ)
                    shutil_copytree(src_dir, dst_dir)
                tags['old/new ({})'.format(organ)] = '{}/{}'.format(old_model_dice, old_model_dice)
        else:
            tags['old/new ({})'.format(organ)] = 'NA/NA'


    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    #       Zip best models and corresponding matrices      #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    zip_file_path = new_model_dir + '.zip'
    zf = zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED)
    logger.info("Zip {} from {}".format(zip_file_path, new_model_dir))
    for root, dirs, files in os.walk(new_model_dir):
        for file_name in files:
            src_file_path = os.path.join(root, file_name)
            dst_file_path = src_file_path.replace(new_model_dir + '/', '')
            zf.write(src_file_path, dst_file_path)
    zf.close()


    # # # # # # # # # # # # # # # # # # # # # # #
    #       Upload to aifs model repository     #
    # # # # # # # # # # # # # # # # # # # # # # #
    if config['production'] == 'retrain_aifs':
        logger.info("Upload to model repository {}".format(default_repo_name))
        afs_models = models()

        afs_models.upload_model(
            model_path=zip_file_path,
            model_repository_name=default_repo_name,
            tags=tags
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--upload", action="store_true")
    args = parser.parse_args()

    if args.download:
        download_from_model_repo()

    if args.upload:
        upload_to_model_repo()