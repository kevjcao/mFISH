#!/usr/bin/env python3

import matplotlib.pyplot as plt
import cellpose
from cellpose import models, io, plot, utils
import tkinter as tk
from tkinter import filedialog
import numpy as np
import os
from skimage import measure
import re
import pandas as pd
from matplotlib import cm

# locate directory for mFISH round & create folder to hold cellpose outputs
root = tk.Tk()
root.withdraw()
image_path = filedialog.askopenfilename()
root = os.path.split(image_path)
cellpose_output = root[0] + '/cellpose_output'
if not os.path.exists(cellpose_output):
    os.makedirs(cellpose_output)

# view raw image
fig1 = seg_img = io.imread(image_path)
plt.figure(figsize=(10, 10))
plt.imshow(seg_img)
plt.axis('off')
plt.title(root[1])
plt.tight_layout()
plt.show()

# read tif into cellpose; channels=[0,0] for grayscale image
model = models.Cellpose(gpu=False, model_type='cyto2')
masks, flows, styles, diams = model.eval(seg_img, diameter=None, channels=[0,0],
                                         flow_threshold=0.4, do_3D=False)

# save cellpose segmentation as _seg.npy
io.masks_flows_to_seg(seg_img, masks, flows, diams, image_path)

# save segmentation as .txt for imageJ; outlines also used for quantification
outlines = utils.outlines_list(masks)
io.outlines_to_text(root[0] + '/cellpose_output/', outlines)

# plot segmentation, mask, and pose outputs
fig2 = plt.figure(figsize=(40, 10.5))
plot.show_segmentation(fig2, seg_img, masks, flows[0], channels=[0, 0])
plt.tight_layout()
seg_file = root[1].replace(".tif", ".png")
plt.savefig(root[0] + '/cellpose_output/' + seg_file)
plt.show()

# generate binary masks
seg_ch_bool = masks !=0
seg_ch_int = seg_ch_bool*1
invert_mask = 1 - seg_ch_int

fig3 = plt.figure(figsize=(20, 10.5))
plt.subplot(1, 2, 1)
plt.imshow(seg_ch_int, cmap='gray')
plt.title('binary mask')
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

fig4 = plt.figure(figsize=(10, 10))
plt.imshow(dat['img'])
for o in outlines:
    plt.plot(o[:, 0], o[:, 1], linewidth=0.5, color='r')
plt.axis('off')
plt.title(root[1] + ' segmentation')
plt.tight_layout()
plt.show()

# prompt to open  images of other channels for segmentation
ch_image_list = filedialog.askopenfilenames()

colors = ['grey', 'red', 'green', 'orange', 'blue', 'yellow', 'purple', 'magenta', '']
bounds = [0,1,2,3,4,5,6,7,8,9,10]

ch_mgvs = {}
ch_thresholds = []
ch_segm = []
for idx, ch_path in enumerate(ch_image_list):
    ch = io.imread(ch_path)
    ch_root = os.path.split(ch_path)
    seg_ch = cellpose.plot.outline_view(ch, masks, color=[255, 0, 0], mode='inner')
    # plt.figure(figsize=(20, 10.5))
    # plt.subplot(1, 2, 1)
    # plt.imshow(ch)
    # plt.axis('off')
    # plt.title(ch_root[1])
    # plt.subplot(1, 2, 2)
    # plt.imshow(seg_ch)
    # plt.axis('off')
    # plt.title('Rn28s segmentation overlay')
    # plt.tight_layout()
    # plt.show()

    # measure cell properties in each ch from cellpose mask
    #labeled_masks, num_labels = measure.label(masks, return_num=True)
    cell_properties = measure.regionprops_table(masks, intensity_image=ch,
                                                properties=['mean_intensity'])
    ch_mgvs[ch_path] = pd.DataFrame(cell_properties)
    ## threshold all cells
    # set threshold: measure avg of non-segmented regions as background
    ch_thresh = np.mean(ch*invert_mask) + np.std(ch*invert_mask)
    ch_thresholds.append((ch_root[1], ch_thresh))
    # python indexing starts at 0
    thresh_indices = np.where(cell_properties['mean_intensity'] >= ch_thresh)
    # mask labels from cellpose start at 1
    valid_masks = np.array(thresh_indices) + 1

    thresh_masks = np.isin(masks, valid_masks)
    ch_segm.append(thresh_masks)

    seg_ch_thresh = cellpose.plot.outline_view(ch, thresh_masks, color=[255, 0, 0], mode='inner')
    # plt.figure(figsize=(20, 10.5))
    # plt.subplot(1, 2, 1)
    # plt.imshow(thresh_masks)
    # plt.axis('off')
    # plt.title(ch_root[1] + ' thresholded mask')
    # plt.subplot(1, 2, 2)
    # plt.imshow(seg_ch_thresh)
    # plt.axis('off')
    # plt.title('overlay')
    # plt.tight_layout()
    # plt.show()

    plt.figure(figsize=(10, 10.5))
    cmap = cm.get_cmap("RdGy", 2)
    plt.imshow(thresh_masks + seg_ch_int, cmap=cm.RdGy)
    plt.axis('off')
    plt.title('Rn28s segmentation overlay')
    plt.tight_layout()
    plt.show()

overlap = seg_ch_int + ch_segm[0]*2 + ch_segm[1]*3 + ch_segm[2]*4
plt.figure(figsize=(10, 10))
cmap = cm.get_cmap("turbo", 10)
plt.imshow(overlap, cmap=cm.turbo)
plt.axis('off')
plt.title('overlay')
plt.colorbar(ticks=np.arange(0,10))
plt.tight_layout()
plt.show()







