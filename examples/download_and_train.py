import t4
import os
import tqdm
import json
import pandas as pd

from fnet.cli.init import save_default_train_options

###################################################
# Download the 3D multi-channel tiffs via Quilt/T4
###################################################

gpu_id = 0

n_images_to_download = 40
train_fraction = 0.75

image_save_dir = '{}/images/'.format(os.getcwd())
model_save_dir = '{}/model/'.format(os.getcwd())
prefs_save_path = '{}/prefs.json'.format(os.getcwd())

data_save_path_train =  '{}/image_list_train.csv'.format(image_save_dir)
data_save_path_test =  '{}/image_list_test.csv'.format(image_save_dir)

if not os.path.exists(image_save_dir):
    os.makedirs(image_save_dir)


aics_pipeline = t4.Package.browse(
    "aics/pipeline",
    registry="s3://quilt-aics"
)

len(aics_pipeline['fov'])

image_ids = [k for k in aics_pipeline['fov']][0:n_images_to_download]

metadata = {}
for image_id in image_ids:
    metadata[image_id] = aics_pipeline['fov'][image_id].get_meta()

image_save_paths = ['{}/{}'.format(image_save_dir, image_id) for image_id in image_ids]    
    
for image_id, image_save_path in zip(image_ids, image_save_paths):
    
    if os.path.exists(image_save_path):
        continue
        
    try: #We only do this because T4 hates our filesystem. It probably wont affect you.
        aics_pipeline['fov'][image_id].fetch()
    except:
        pass
    
###################################################
# Make a manifest of all of the files in csv form
###################################################



df = pd.DataFrame(columns = ['path_tiff','channel_signal', 'channel_target'])

df['path_tiff'] = image_save_paths

df['channel_signal'] = [metadata[image_id]['content_info']['brighfield_channel'] for image_id in image_ids]
df['channel_target'] = [metadata[image_id]['content_info']['dna_channel'] for image_id in image_ids]

n_train_images = int(n_images_to_download*train_fraction)
df_train = df[0:n_train_images]
df_test = df[n_train_images:]


df_test.to_csv(data_save_path_test, index=False)
df_train.to_csv(data_save_path_train, index=False)

################################################
# Run the label-free stuff (dont change this)
################################################


save_default_train_options(prefs_save_path)

with open(prefs_save_path, 'r') as fp:
    prefs = json.load(fp)
    
    
prefs['dataset_train'] = 'fnet.data.AICSTiffDataset'
prefs['dataset_train_kwargs'] = {'path_csv': data_save_path_train}
prefs['dataset_val'] = 'fnet.data.AICSTiffDataset'
prefs['dataset_val_kwargs'] = {'path_csv': data_save_path_test}


with open(prefs_save_path, 'w') as fp:
    json.dump(prefs, fp)

command_str = 'python ../fnet/cli/train_model.py {} --gpu_ids {}'.format(prefs_save_path, gpu_id)

print(command_str)
os.system(command_str)