# Merge images into class-specific folders while retaining well and depth identifiers.
from PIL import Image
import os

# Use project-relative directory paths rather than machine-specific Windows paths.
outer_path = os.environ.get('FC_SVNET_RESIZE_DIR', os.path.join('data', 'resized'))
output_root = os.environ.get('FC_SVNET_MERGE_DIR', os.path.join('data', 'merged'))
folderlist = os.listdir(outer_path)  # List well-specific directories.
print(folderlist)
total_num_folder = len(folderlist)  # Number of well directories.
print('total have %d folders' % (total_num_folder))  # Print well count.

for folder0 in folderlist:# Iterate over well identifiers.
    mid_path = os.path.join(outer_path, folder0)  # Directory for the current well.
    folderlist1 = os.listdir(mid_path)# List numeric class directories.
    total_num_folder1 = len(folderlist1)  # Number of classes.
    print('{} total have {} folders'.format(folder0,total_num_folder1))
    print(folderlist1)
    for folder in folderlist1:# Iterate over class directories.
        inner_path = os.path.join(mid_path, folder)# Directory for the current class.
        filelist = os.listdir(inner_path)  # List image files.
        total_num_file = len(filelist)  # Number of image files.
        print('{} have {} file'.format(folder,total_num_file))  # Print image count.
        # Create the output class directory.
        dir = os.path.join(output_root, str(folder))
        if not os.path.exists(dir):
            # Create output folders as needed.
            os.makedirs(dir, exist_ok=True)
        for item in filelist:# Iterate over images.
            if item.endswith('.jpg'):
                src = os.path.join(os.path.abspath(inner_path), item)  # Source image path.
                # Destination image path.
                dst = os.path.join(output_root, str(folder), str(item))
                img = Image.open(src)
                out = img.resize((512, 512), Image.HAMMING)  # Resize to 512 x 512 using Hamming resampling.
                out.save(dst)
