# Resize images in batches, preserving well and depth information.
from PIL import Image
import os

# Use a platform-independent input directory.
outer_path = os.environ.get('FC_SVNET_CROP_DIR', os.path.join('data', 'cropped'))  # Directory containing cropped images.
folderlist = os.listdir(outer_path)  # List well directories.
print(folderlist)
total_num_folder = len(folderlist)  # Number of wells.
print('total have %d folders' % (total_num_folder))  # Print well count.

for folder0 in folderlist:  # Iterate over wells.
    mid_path = os.path.join(outer_path, folder0)  # Access one well directory.
    folderlist1 = os.listdir(mid_path)  # List numeric class directories.
    total_num_folder1 = len(folderlist1)  # Number of class directories.
    print('{} total have {} folders'.format(folder0, total_num_folder1))
    print(folderlist1)

    for folder in folderlist1:
        inner_path = os.path.join(mid_path, folder)  # Access one numeric class directory.
        filelist = os.listdir(inner_path)  # List images.
        total_num_file = len(filelist)  # Number of images.
        print('{} have {} file'.format(folder, total_num_file))  # Print the image count.

        # Build output paths using os.path.join.
        save_root = os.environ.get('FC_SVNET_RESIZE_DIR', os.path.join('data', 'resized'))
        dir = os.path.join(save_root, folder0)  # Output directory for this well.
        if not os.path.exists(dir):
            os.makedirs(dir, exist_ok=True)

        dir = os.path.join(save_root, folder0, folder)  # Output directory for this well and class.
        if not os.path.exists(dir):
            os.makedirs(dir, exist_ok=True)

        for item in filelist:
            if item.endswith('.jpg'):
                src = os.path.join(os.path.abspath(inner_path), item)  # Source image path.
                dst = os.path.join(save_root, folder0, folder, item)  # Destination image path.
                img = Image.open(src)
                out = img.resize((224, 224), Image.HAMMING)  # Resize with the Hamming resampling filter.
                out.save(dst)
