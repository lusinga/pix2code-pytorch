import os
import random
from tqdm import tqdm
from shutil import copyfile

raw_data_dir = '../pix2code/datasets/android/all_data/'
assert os.path.isdir(raw_data_dir)

output_dir = './processed_data/'

filenames = os.listdir(raw_data_dir)

# Transform filenames into pairs (.gui, .png)
filenames = [(f[:-3] + 'gui', f[:-3] + 'png') for f in filenames if f.endswith('.gui')]

random.seed(12345)
filenames.sort()
random.shuffle(filenames)

split_1 = int(0.8 * len(filenames))
split_2 = int(0.9 * len(filenames))

filenames = {
    'train': filenames[:split_1],
    'dev': filenames[split_1:split_2],
    'test': filenames[split_2:]
}

if not os.path.exists(output_dir):
    os.mkdir(output_dir)
else:
    print('Warning: output dir {} already exists.'.format(output_dir))

for split in ['train', 'dev', 'test']:
    output_dir_split = os.path.join(output_dir, 'data_{}'.format(split))
    
    if not os.path.exists(output_dir_split):
        os.mkdir(output_dir_split)
    else:
        print('Warning: output dir {} already exists.'.format(output_dir_split))
        
    print('Processing {} data, saving to {}.'.format(split, output_dir_split))
    
    for (gui, png) in tqdm(filenames[split]):
        src_path_gui = os.path.join(raw_data_dir, gui)
        output_path_gui = os.path.join(output_dir_split, gui)
        src_path_png = os.path.join(raw_data_dir, png)
        output_path_png = os.path.join(output_dir_split, png)
        
        copyfile(src_path_gui, output_path_gui)
        copyfile(src_path_png, output_path_png)
