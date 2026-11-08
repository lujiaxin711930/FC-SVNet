# Generate one well-log image from each group of six samples (0.25 m, no sliding window); retain well and depth identifiers.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import os
import re

# Disable automatic scientific-notation offsets on the axes.
plt.rcParams['axes.formatter.useoffset'] = False

outer_path = os.environ.get('FC_SVNET_EXCEL_DIR', os.path.join('data', 'labeled_logs'))
output_root = os.environ.get('FC_SVNET_PLOT_DIR', os.path.join('data', 'plotted'))


def clean_path_name(name):
    if isinstance(name, (int, float, np.floating)):
        name = str(name)
    name = re.sub(r'[\0\x00-\x1f\x7f]', '', name)
    name = re.sub(r'[<>:"|?*\\/]', '_', name)
    return name


folderlist = os.listdir(outer_path)
print(folderlist)
print(f'total have {len(folderlist)} folders')

# Sampling tolerance and depth-window settings.
WIN_POINTS = 6  # Six samples per image.
WIN_DEPTH = 0.25  # Target depth span in meters.
DEPTH_TOL = 1e-6  # Floating-point comparison tolerance.

for folder in folderlist:
    print(f'Processing {folder}:')
    file_name = clean_path_name(os.path.splitext(folder)[0])

    excel_path = os.path.join(outer_path, folder)
    df = pd.read_excel(excel_path)

    df = df.replace({'岩相': {'St': 0, 'Sa': 1, 'Sp': 2, 'Sw': 3,
                              'Fw': 4, 'Fh': 5, 'O': 6}})

    df.loc[:, 'GR'] = df.loc[:, "GR"] - 39.815
    df.loc[:, 'AC'] = df.loc[:, "AC"] - 61.917 + 151.279 + 50
    df.loc[:, 'LLD'] = df.loc[:, "LLD"] - 1.726 + 643.2039999 + 50
    df.loc[:, 'LLS'] = df.loc[:, "LLS"] - 1.522 + 832.1329999 + 50
    df.loc[:, 'SP'] = df.loc[:, "SP"] - 44.943 + 983.5859999 + 50
    df.loc[:, 'CAL'] = df.loc[:, "CAL"] - 20.204 + 1057.6377999 + 50

    data = df.values
    biaoqian = data[:, 7]
    deep = data[:, 0].astype(float)
    n = len(df)

    if n > 1:
        med_step = float(np.median(np.abs(np.diff(deep))))
        if not (0.03 < med_step < 0.07):
            print(f'  [WARNING] {file_name}: median depth increment = {med_step:.4f} m; expected approximately 0.05 m')

    i = 0
    count = 1

    while i + WIN_POINTS <= n:
        deepsum = deep[i + WIN_POINTS - 1] - deep[i]
        if 0 < deepsum <= WIN_DEPTH + DEPTH_TOL:
            flaglist = [biaoqian[i + k] for k in range(WIN_POINTS)]
            counts = {}
            for v in flaglist:
                counts[v] = counts.get(v, 0) + 1
            max_cnt = max(counts.values())
            candidates = [v for v, c in counts.items() if c == max_cnt]
            try:
                flag = int(candidates[0])
            except (TypeError, ValueError):
                print(f'  [SKIP] Invalid integer label at depth {deep[i]}: {candidates[0]!r}')
                i += WIN_POINTS
                continue

            output_dir = os.path.join(output_root, file_name, str(flag))
            os.makedirs(output_dir, exist_ok=True)

            j = i + WIN_POINTS
            x0 = deep[i:j]
            x1 = data[i:j, 1].astype(float)
            x2 = data[i:j, 2].astype(float)
            x3 = data[i:j, 3].astype(float)
            x4 = data[i:j, 4].astype(float)
            x5 = data[i:j, 5].astype(float)
            x6 = data[i:j, 6].astype(float)

            # Use a fixed initial figure size.
            # At 100 dpi, a 5.74 x 4.28-inch figure corresponds to 574 x 428 pixels before saving.
            plt.figure(figsize=(5.74, 4.28), dpi=100)

            plt.plot(x1, x0, c='r', linewidth=2.5)
            plt.plot(x2, x0, c='b', linewidth=2.5)
            plt.plot(x3, x0, c='g', linewidth=2.5)
            plt.plot(x4, x0, c='m', linewidth=2.5)
            plt.plot(x5, x0, c='y', linewidth=2.5)
            plt.plot(x6, x0, c='c', linewidth=2.5)

            plt.gca().invert_yaxis()
            ax = plt.gca()
            ax.xaxis.set_major_locator(MultipleLocator(50))
            plt.xlim(0, 1200)

            # Disable the y-axis offset formatter.
            ax.yaxis.get_major_formatter().set_useOffset(False)

            img_name = f'{file_name}_{clean_path_name(deep[i])}_{count}.jpg'
            img_path = os.path.join(output_dir, img_name)

            # Save the figure using the original tight-bounding-box option.
            # Note: bbox_inches="tight" and dpi=600 can change the final pixel size.
            plt.savefig(img_path, bbox_inches='tight', dpi=600)

            plt.close()
            count += 1
            i += WIN_POINTS
        else:
            i += 1

print('Completed')
