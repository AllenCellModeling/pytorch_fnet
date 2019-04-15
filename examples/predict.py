import os

###################################################
# Assume the user already ran download_and_train.py
###################################################

gpu_id = 0

image_save_dir = '{}/images/'.format(os.getcwd())
model_save_dir = '{}/model/'.format(os.getcwd())
prefs_save_path = '{}/prefs.json'.format(os.getcwd())

data_save_path_test = '{}/image_list_test.csv'.format(image_save_dir)

command_str = 'fnet predict ./ ' \
    '--dataset fnet.data.MultiChTiffDataset ' \
    '--dataset_kwargs \'{{"path_csv": "{}"}}\' ' \
    '--gpu_ids {}'.format(data_save_path_test, gpu_id)

print(command_str)
os.system(command_str)
