import pickle
import cv2
from scipy.ndimage.measurements import label
from scipy.ndimage.morphology import generate_binary_structure
from vehicletracker import VehicleTracker
from utils import *

dist_pickle = pickle.load( open("svc_pickle.p", "rb" ) )
svc = dist_pickle["svc"]
X_scaler = dist_pickle["scaler"]
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
spatial_size = dist_pickle["spatial_size"]
hist_bins = dist_pickle["hist_bins"]

vehicleTracker = VehicleTracker()

def pipeline(image):

    # define search areas and scales
    scales            = [2.5,         2,           1.5,          1.25,        1,           .5]
    x_start_stop_list = [[400, 1280], [384, 1280], [480, 1280], [400, 1280],  [624, 1024], [624, 896]]
    y_start_stop_list = [[500, 700],  [400, 656],  [400, 600],  [375, 520],   [400, 480],  [400, 450]]

    box_list = []
    # detect cars
    for y_start_stop, x_start_stop, scale in zip(y_start_stop_list, x_start_stop_list, scales):
        box = find_cars(image, y_start_stop[0], y_start_stop[1], x_start_stop[0], x_start_stop[1], scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
        box_list.extend(box)

    out_img = draw_boxes(np.copy(image), box_list)

    heat = np.zeros_like(image[:,:,0]).astype(np.float)
    # Add heat to each box in box list
    heat = add_heat(heat, box_list)

    vehicleTracker.heatmaps.append(heat)
    heat = np.sum(vehicleTracker.heatmaps, axis=0)

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, vehicleTracker.threshold)

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)
    # Find final boxes from heatmap using label function

    structure = [[1, 1, 1],
                 [1, 1, 1],
                 [1, 1, 1]]

    labels = label(heatmap, structure=structure)

    draw_img = draw_labeled_bboxes(np.copy(image), labels)

    image_overlay(out_img, draw_img, 1.0, 0.25, origin=(0, 0))

    heatmap = cv2.equalizeHist(heatmap.astype(np.uint8))
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, code=cv2.COLOR_BGR2RGB)

    image_overlay(heatmap, draw_img, 1.0, 0.25, origin=(int(image.shape[1] * 0.25), 0))

    return draw_img
