#!/usr/bin/env python3
import os, sys
import yaml
from collections import OrderedDict
from filelock import FileLock
import argparse
import zipfile
import time
import glob

current_dir = os.path.dirname(os.path.abspath(__file__))
monai_dir = os.path.dirname(current_dir)

# https://stackoverflow.com/questions/5121931/in-python-how-can-you-load-yaml-mappings-as-ordereddicts
def ordered_load(stream, Loader=yaml.Loader, object_pairs_hook=OrderedDict):
    class OrderedLoader(Loader):
        pass
    def construct_mapping(loader, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader.construct_pairs(node))
    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping)
    return yaml.load(stream, OrderedLoader)

# https://ttl255.com/yaml-anchors-and-aliases-and-how-to-disable-them/
def ordered_dump(data, stream=None, Dumper=yaml.Dumper, **kwds):
    class OrderedDumper(Dumper):
        def ignore_aliases(self, data):
            return True
    def _dict_representer(dumper, data):
        return dumper.represent_mapping(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
            data.items())
    OrderedDumper.add_representer(OrderedDict, _dict_representer)
    return yaml.dump(data, stream, OrderedDumper, **kwds)

def read_config_yaml(config_file):
    config_file_lock = config_file + ".lock"
    lock = FileLock(config_file_lock, timeout=30)

    if os.path.isfile(config_file):
        with lock:
            with open(config_file) as file:
                # config_dict = yaml.load(file, Loader=yaml.Loader)
                config_dict = ordered_load(file, yaml.SafeLoader)
                if config_dict==None:
                    config_dict = OrderedDict()
    else:
        config_dict = OrderedDict()
    return config_dict

def write_config_yaml(config_file, write_dict):
    config_file_lock = config_file + ".lock"
    lock = FileLock(config_file_lock, timeout=30)

    config_dict = read_config_yaml(config_file)
    config_dict.update(write_dict)
    # for key, value in write_dict.items():
    #     config_dict[key] = value

    with lock:
        with open(config_file, 'w') as file:
            # yaml.dump(config_dict, file, default_flow_style=False)
            ordered_dump(config_dict, file, Dumper=yaml.SafeDumper, default_flow_style=False)

def write_config_yaml_with_key_value(config_file, key, value):
    config_file_lock = config_file + ".lock"
    lock = FileLock(config_file_lock, timeout=30)

    config_dict = read_config_yaml(config_file)
    config_dict[key] = value

    with lock:
        with open(config_file, 'w') as file:
            # yaml.dump(config_dict, file, default_flow_style=False)
            ordered_dump(config_dict, file, Dumper=yaml.SafeDumper, default_flow_style=False)

def print_config_yaml(config_file):
    config_dict = read_config_yaml(config_file)
    print(dict(config_dict))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--set_production", type=str)
    args = parser.parse_args()

    # Read config_file
    config_file = os.path.join(current_dir, 'config.yaml')
    config_dict = read_config_yaml(config_file)

    if args.set_production:
        config_dict['production'] = args.set_production # retrain / retrain_aifs / inference / inference_aifs
        subdict = {'production': args.set_production}
        print(subdict)
    else:
        config_dict['train_data_dir'] = os.path.join(monai_dir, 'train_data')
        config_dict['val_data_dir'] = os.path.join(monai_dir, 'val_data')
        config_dict['test_data_dir'] = os.path.join(monai_dir, 'test_data')
        config_dict['models_dir'] = os.path.join(monai_dir, 'models')
        # config_dict['old_model_dir'] # model_repo_manip.py update this in retrain or retrain_aifs mode: os.path.join(monai_dir, 'models/old')
        config_dict['organ_list'] = None # train.py or inference_api.py update this in all modes: ['liver', 'pancreas', 'spleen']
        config_dict['roi_size'] = (256, 256, 16)

        # https://catalog.ngc.nvidia.com/orgs/nvidia/teams/med/models/clara_pt_liver_and_tumor_ct_segmentation
        # https://catalog.ngc.nvidia.com/orgs/nvidia/teams/med/models/clara_pt_pancreas_and_tumor_ct_segmentation
        # https://catalog.ngc.nvidia.com/orgs/nvidia/teams/med/models/clara_pt_spleen_ct_segmentation
        if config_dict['production'] == 'retrain' or config_dict['production'] == 'retrain_aifs':
            today_date = time.strftime("%Y_%m_%d")
            config_dict['organ_to_mmar'] = \
                {
                    'liver'   : OrderedDict({
                                'name': 'clara_pt_liver_and_tumor_ct_segmentation_1',
                                'channel': 3,
                                'mmar_dir': os.path.join(monai_dir, 'models/mmar/liver'),
                                'old_model_dir': os.path.join(config_dict['old_model_dir'], 'liver') if config_dict['old_model_dir'] else None,
                                'old_model_file_path': None,
                                'old_model_dice': None,
                                'new_model_dir': os.path.join(monai_dir, 'models/{}/liver'.format(today_date)),
                                'new_model_file_path': None,
                                'new_model_dice': None
                                }),
                    'pancreas': OrderedDict({
                                'name': 'clara_pt_pancreas_and_tumor_ct_segmentation_1',
                                'channel': 3,
                                'mmar_dir': os.path.join(monai_dir, 'models/mmar/pancreas'),
                                'old_model_dir': os.path.join(config_dict['old_model_dir'], 'pancreas') if config_dict['old_model_dir'] else None,
                                'old_model_file_path': None,
                                'old_model_dice': None,
                                'new_model_dir': os.path.join(monai_dir, 'models/{}/pancreas'.format(today_date)),
                                'new_model_file_path': None,
                                'new_model_dice': None
                                }),
                    'spleen'  : OrderedDict({
                                'name': 'clara_pt_spleen_ct_segmentation_1',
                                'channel': 2,
                                'mmar_dir': os.path.join(monai_dir, 'models/mmar/spleen'),
                                'old_model_dir': os.path.join(config_dict['old_model_dir'], 'spleen') if config_dict['old_model_dir'] else None,
                                'old_model_file_path': None,
                                'old_model_dice': None,
                                'new_model_dir': os.path.join(monai_dir, 'models/{}/spleen'.format(today_date)),
                                'new_model_file_path': None,
                                'new_model_dice': None
                                })
                }

            for organ in config_dict['organ_to_mmar']:
                old_model_dir = config_dict['organ_to_mmar'][organ]['old_model_dir']
                if old_model_dir != None:
                    old_model_file_path_list = glob.glob(os.path.join(old_model_dir, "*.pth"))
                    if len(old_model_file_path_list) > 0:
                        config_dict['organ_to_mmar'][organ]['old_model_file_path'] = old_model_file_path_list[0]
        else:
            config_dict['organ_to_mmar'] = \
                {
                    'liver'   : OrderedDict({
                                'name': 'clara_pt_liver_and_tumor_ct_segmentation_1',
                                'channel': 3,
                                'mmar_dir': os.path.join(monai_dir, 'models/mmar/liver'),
                                'old_model_dir': None,
                                'old_model_file_path': None,
                                'old_model_dice': None,
                                'new_model_dir': os.path.join(monai_dir, 'models/inference/liver'),
                                'new_model_file_path': None,
                                'new_model_dice': None
                                }),
                    'pancreas': OrderedDict({
                                'name': 'clara_pt_pancreas_and_tumor_ct_segmentation_1',
                                'channel': 3,
                                'mmar_dir': os.path.join(monai_dir, 'models/mmar/pancreas'),
                                'old_model_dir': None,
                                'old_model_file_path': None,
                                'old_model_dice': None,
                                'new_model_dir': os.path.join(monai_dir, 'models/inference/pancreas'),
                                'new_model_file_path': None,
                                'new_model_dice': None
                                }),
                    'spleen'  : OrderedDict({
                                'name': 'clara_pt_spleen_ct_segmentation_1',
                                'channel': 2,
                                'mmar_dir': os.path.join(monai_dir, 'models/mmar/spleen'),
                                'old_model_dir': None,
                                'old_model_file_path': None,
                                'old_model_dice': None,
                                'new_model_dir': os.path.join(monai_dir, 'models/inference/spleen'),
                                'new_model_file_path': None,
                                'new_model_dice': None
                                })
                }

            for organ in config_dict['organ_to_mmar']:
                new_model_dir = config_dict['organ_to_mmar'][organ]['new_model_dir']
                if new_model_dir != None:
                    new_model_file_path_list = glob.glob(os.path.join(new_model_dir, "*.pth"))
                    if len(new_model_file_path_list) > 0:
                        config_dict['organ_to_mmar'][organ]['new_model_file_path'] = new_model_file_path_list[0]

    # Write config_file
    write_config_yaml(config_file, config_dict)