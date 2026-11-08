# Crop images for each well while preserving well and depth information in filenames.
from PIL import Image
import os

# Input directory for plotted images.
outer_path = os.environ.get('FC_SVNET_PLOT_DIR', os.path.join('data', 'plotted'))
# Output paths are configured relative to the project by default.
output_root = os.environ.get('FC_SVNET_CROP_DIR', os.path.join('data', 'cropped'))

folderlist = os.listdir(outer_path)  # List the well-specific input directories.
print(folderlist)
total_num_folder = len(folderlist)  # Number of well directories.
print('total have %d folders' % (total_num_folder))  # Print the number of wells.

for folder0 in folderlist:  # Iterate through the well directories.
    mid_path = os.path.join(outer_path, folder0)  # Access each well directory.
    folderlist1 = os.listdir(mid_path)  # List class directories (0-6).
    total_num_folder1 = len(folderlist1)  # Count class directories.
    print('{} total have {} folders'.format(folder0, total_num_folder1))
    print(folderlist1)
    
    for folder in folderlist1:  # Iterate over the seven numeric classes.
        inner_path = os.path.join(mid_path, folder)  # Access one class directory.
        filelist = os.listdir(inner_path)  # List images.
        total_num_file = len(filelist)  # Count images in this directory.
        print('{} have {} file'.format(folder, total_num_file))  # Print image count.
        
        # Create the destination path with os.path.join.
        output_folder = os.path.join(output_root, folder0, folder)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)  # Create nested destination directories as needed.
        
        for item in filelist:
            if item.endswith('.jpg'):
                src = os.path.join(inner_path, item)  # Source image path.
                dst = os.path.join(output_folder, item)  # Destination image path.
                
                img = Image.open(src)
                # Crop coordinates: (left, top, right, bottom).
                # Alternative crop coordinates used in an earlier experiment.
                cropped = img.crop((465, 120, 2993, 1969))
                cropped.save(dst)
