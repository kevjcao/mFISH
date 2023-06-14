#!/usr/bin/env python3
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
import os
import re
import cellpose
from cellpose import models, io, plot, utils
from tkinter import filedialog
from skimage import measure
from matplotlib import cm, patches, colors
from matplotlib.colors import ListedColormap
from scipy.ndimage import rotate


# locate directory for mFISH round & create folder to hold cellpose outputs
root = tk.Tk()
root.withdraw()
image_path = filedialog.askopenfilename()
root = os.path.split(image_path)
cellpose_output = root[0] + '/cellpose_output'
if not os.path.exists(cellpose_output):
    os.makedirs(cellpose_output)
seg_img = io.imread(image_path)

# read tif into cellpose; channels=[0,0] for grayscale image
model = models.Cellpose(gpu=False, model_type='cyto2')
masks, flows, styles, diams = model.eval(seg_img, diameter=None, channels=[0,0],
                                         flow_threshold=0.4, do_3D=False)
# save cellpose segmentation as _seg.npy
io.masks_flows_to_seg(seg_img, masks, flows, diams, image_path)
# save segmentation as .txt for imageJ; outlines also used for quantification
outlines = utils.outlines_list(masks)
io.outlines_to_text(root[0] + '/cellpose_output/', outlines)

fig1 = plt.figure(figsize=(40, 10.5))
plot.show_segmentation(fig1, seg_img, masks, flows[0], channels=[0, 0])
plt.tight_layout()
seg_file = root[1].replace(".tif", ".png")
plt.savefig(root[0] + '/cellpose_output/' + seg_file)
plt.show()

masks_prop = measure.regionprops_table(masks, intensity_image=masks,
                                                properties=['centroid'])
# raster_pts = np.vstack((masks_prop['centroid-1'], masks_prop['centroid-0'])).T

# generate binary masks
seg_ch_bool = masks !=0
seg_ch_int = seg_ch_bool*1
invert_mask = 1 - seg_ch_int

fig2 = plt.figure(figsize=(20, 10.5))
plt.subplot(1, 2, 1)
plt.imshow(seg_ch_int, cmap='gray')
plt.title('Rn28s binary mask')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(invert_mask, cmap='gray')
plt.title('inverted mask')
plt.axis('off')
plt.tight_layout()
plt.show()

# obtain ROIs for each segmented cell; plot red overlay on original image
np_seg = re.sub('.tif$', '_seg.npy', image_path)
dat = np.load(np_seg, allow_pickle=True).item()
outlines = utils.outlines_list(dat['masks'])

fig3 = plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.imshow(seg_img)
plt.axis('off')
plt.title(root[1])
plt.subplot(1, 2, 2)
plt.imshow(dat['img'])
for o in outlines:
    plt.plot(o[:, 0], o[:, 1], linewidth=0.5, color='r')
plt.axis('off')
plt.title(root[1] + ' segmentation')
plt.tight_layout()
plt.show()

# prompt to open  images of other channels for segmentation
ch_image_list = filedialog.askopenfilenames()

ch_mgvs = {}
ch_thresholds = []
mask_list = []
for idx, ch_path in enumerate(ch_image_list):
    ch = io.imread(ch_path)
    ch_root = os.path.split(ch_path)
    # measure cell properties in each ch from Rn28s cellpose mask
    labeled_masks, num_labels = measure.label(masks, return_num=True)
    cell_properties = measure.regionprops_table(masks, intensity_image=ch,
                                                properties=['mean_intensity', 'centroid'])
    ch_mgvs[ch_path] = pd.DataFrame(cell_properties)
    # set threshold: measure avg of non-segmented regions as background
    ch_thresh = np.mean(ch*invert_mask) + 1.25*np.std(ch*invert_mask)
    ch_thresholds.append((ch_root[1], ch_thresh))
    # python indexing starts at 0
    thresh_indices = np.where(cell_properties['mean_intensity'] >= ch_thresh)
    # mask labels from cellpose start at 1
    valid_masks = np.array(thresh_indices) + 1
    thresh_masks = np.isin(masks, valid_masks)
    #cell_properties = measure.regionprops_table(thresh_masks, intensity_image=ch,
                                                #properties=['mean_intensity', 'centroid'])
    mask_list.append((ch_root[1], thresh_masks))
    seg_ch_thresh = cellpose.plot.outline_view(ch, thresh_masks, color=[255, 0, 0], mode='inner')
    #
    # plt.figure(figsize=(20, 20.5))
    # plt.subplot(2, 2, 1)
    # # plt.imshow(ch)
    # plt.imshow(thresh_masks)
    # plt.axis('off')
    # plt.title(ch_root[1] + ' masks')
    # #plt.title(ch_root[1] + 'raw')
    # plt.subplot(2, 2, 2)
    # plt.imshow(seg_ch_thresh)
    # plt.axis('off')
    # plt.title('mask overlay')
    # plt.subplot(2, 2, 3)
    # cmap = cm.get_cmap("viridis", 2)
    # plt.imshow(thresh_masks + seg_ch_int, cmap=cm.viridis)
    # plt.axis('off')
    # plt.title(ch_root[1] + ' Rn28s segmentation overlay')
    # plt.subplot(2, 2, 4)
    # plt.scatter(cell_properties['centroid-1'], cell_properties['centroid-0'], s=10)
    # plt.gca().invert_yaxis()
    # plt.axis('off')
    # plt.title('Rn28s raster')
    # plt.tight_layout()
    # plt.show()

    plt.figure(figsize=(10, 10.5))
    plt.imshow(thresh_masks + seg_ch_int, cmap='Greys', vmin=0, vmax=np.max(thresh_masks + seg_ch_int))
    plt.axis('off')
    plt.title(ch_root[1] + ' Rn28s segmentation overlay')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 10.5))
    plt.imshow(thresh_masks + seg_ch_int, cmap='Greys', vmin=0, vmax=np.max(thresh_masks + seg_ch_int))
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# cell-type color map based on sum of boolean masks
overlay_img = seg_ch_int
for idx, ch_mask in enumerate(mask_list):
    overlay_img = overlay_img + (ch_mask[1]*(idx+2)*2)

color_dict = {
    0: 'white',         # background
    1: 'gainsboro',     # Rn28s
    2: 'gainsboro',
    3: 'gainsboro',
    4: 'gainsboro',
    5: 'gainsboro',       # gene 1, idx 0
    6: 'gainsboro',
    7: 'gainsboro',  # gene 2, idx 1
    8: 'gainsboro',
    9: 'gainsboro',   # gene 3, idx 2
    10: 'gainsboro',
    11: 'cyan',         # gene 1+2
    12: 'gainsboro',
    13: 'lightcoral',   # gene 1+3
    14: 'gainsboro',
    15: 'gold',         # gene 2+3
    16: 'gold',
    17: 'gainsboro',
    18: 'gainsboro',
    19: 'red',          # gene 1+2+3
}

color_list = [0, 1, 4, 5, 6, 7, 8, 9, 11, 13, 15, 19]
norm = colors.Normalize(vmin=min(color_list), vmax=max(color_list), clip=True)
color_mapper = cm.ScalarMappable(norm=norm, cmap=cm.Greys_r)

# Create a color map based on the dictionary
cmap_over = colors.ListedColormap(list(color_dict.values()))

plt.figure(figsize=(10, 10.5))
plt.imshow(overlay_img, cmap=cmap_over, vmin=0, vmax=np.max(overlay_img))
plt.axis('off')
plt.tight_layout()
plt.show()

# colormap_img = cmap(overlay_img)
plt.figure(figsize=(21, 10.5))
plt.subplot(1, 2, 1)
plt.imshow(overlay_img, cmap='viridis', vmin=0, vmax=np.max(overlay_img))
plt.title('viridis overlay')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(overlay_img, cmap=cmap)
plt.axis('off')
plt.title('gene mask overlay, summation-based')
# legend_elements = [patches.Patch(color=color, label=str(value)) for value, color in zip(pixel_intensities, gene_color) if value !=0]
# legend = plt.legend(handles=legend_elements, bbox_to_anchor=(1, 1), loc='upper left')
# plt.setp(legend.get_title(), fontsize='large')
# plt.subplots_adjust(right=0.5)
plt.tight_layout()
plt.show()


# plt.figure(figsize=(5, 5.2))
# test_raster = plt.scatter(masks_prop['centroid-1'], masks_prop['centroid-0'], s=10)
# plt.gca().invert_yaxis()
# plt.axis('off')
# plt.title('Rn28s raster')
# plt.show()
#
# test_rot = rotate(test_raster, 50)
# plt.imshow(test_rot)
# plt.show()


## manual color map generation
# 1 gene

color_dict = {
    0: 'white',         # background
    1: 'gainsboro',     # Rn28s
    3: 'yellowgreen',
}
cmap = colors.ListedColormap(list(color_dict.values()))
plt.figure(figsize=(10, 10.5))
plt.imshow(mask_list[0][1] + seg_ch_int, cmap=cmap, vmin=0, vmax=np.max(mask_list[0][1] + seg_ch_int))
plt.axis('off')
#plt.title(ch_root[1] + ' Rn28s segmentation overlay')
plt.tight_layout()
plt.show()

# 2 genes
color_dict2 = {
    0: 'white',         # background
    1: 'gainsboro',     # Rn28s
    2: 'yellowgreen',         #gene 1
    3: 'teal',        #gene 2
    4: 'black',         #gene 1+2
}
cmap2 = colors.ListedColormap(list(color_dict2.values()))
plt.figure(figsize=(10, 10.5))
plt.imshow(mask_list[0][1] + (mask_list[1][1]*(2)) + seg_ch_int, cmap=cmap2, vmin=0, vmax=np.max(mask_list[0][1]+ (mask_list[1][1]*(2)) + seg_ch_int))
plt.axis('off')
#plt.title(ch_root[1] + ' Rn28s segmentation overlay')
plt.tight_layout()
plt.show()

