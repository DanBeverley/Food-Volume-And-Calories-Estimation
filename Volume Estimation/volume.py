import numpy as np
import cv2
import os
import json
import glob
from PIL import Image, ImageDraw

plate_diameter  = 25  #cm
plate_depth     = 1.5 #cm
plate_thickness = 0.2 #cm

def Max(x,y):
    """Return maximum of two values
       Called in cal_volume to ensure depth value do not go below 0"""
    if (x>=y):
        return x
    else:
        return y

def polygons_to_mask(img_shape, polygons):
    """Create binary mask from a list of polygon points
    Converting polygonal regions defined in JSON to mask arrays,
    which are then used to calculate bounding boxes and area for each region."""
    mask = np.zeros(img_shape, dtype=np.uint8)
    mask = Image.fromarray(mask)
    xy   = list(map(tuple, polygons))
    ImageDraw.Draw(mask).polygon(xy = xy, outline = 1, fill = 1)
    mask = np.array(mask, dtype = bool)
    return mask

def mask2box(mask):
    """Find the bounding box of the object within the mask
    Used in get_bbox to calculate the minimal rectangular area that contains the polygon points,
    which is later used to speed up the volume calculation by limiting it to the bounding box."""
    index = np.argwhere(mask == 1)
    rows  = index[:,0]
    clos  = index[:,1]
    left_top_r = np.min(rows)
    left_top_c = np.min(clos)
    right_bottom_r = np.max(rows)
    right_bottom_c = np.max(clos)
    return [left_top_c, left_top_r, right_bottom_c, right_bottom_r]

def get_bbox(points, h, w):
    """To get the bounding box for a given polygon region defined by points.
    Calculates the smallest bounding box containing the specified polygon in the depth image.
    This reduces the computational area for volume calculation."""
    polygons = points
    mask     = polygons_to_mask([h,w], polygons)
    return mask2box(mask)

def get_scale(points, img, lowest):
    """Determine the scale for converting pixels to real-world dimensions
    len_per_pix: Real-world length represented by each pixel based on the diameter of the plate.
    depth_per_pix: Real-world depth per pixel, calculated based on the plate’s depth."""
    bbox        = get_bbox(points, img.shape[0], img.shape[1])
    diameter    = (bbox[2]-bbox[0] + 1+ bbox[3]-bbox[1]+1)/2
    len_per_pix = plate_diameter / float(diameter)
    avg         = 0
    k           = 0
    for point in points:
        avg += img[point[1]][point[0]]
        k   += 1
    avg   = avg/float(k)
    depth = lowest - avg
    depth_per_pix = plate_depth/depth

    return len_per_pix, depth_per_pix

def cal_volume(points, img, len_per_pix, depth_per_pix, lowest):
    """ To compute the volume of material contained within a specific polygonal region.
    Steps :
    1.Get the bounding box to limit the area of volume calculation.
    2.Use cv2.pointPolygonTest to check if each pixel inside the bounding box lies within the polygon.
    3.For each pixel inside the polygon, calculate the volume contribution based on pixel depth.
    """
    volume = 0.0
    bbox   = get_bbox(points, img.shape[0], img.shape[1])
    points = np.array(points)
    shape  = points.shape
    points = points.reshape(shape[0], 1, shape[1])
    for i in range(bbox[0], bbox[2]+1):
        for j in range(bbox[1], bbox[3]+1):
            if (cv2.pointPolygonTest(points, (i,j), False)>=0):
                volume += Max(0, (lowest - img[j][i])* depth_per_pix - plate_thickness)*len_per_pix*len_per_pix
    return volume

def get_volume(img, json_path):
    """ To calculate the volume for each labeled object within the image.
    Steps:
    1.Load image and JSON file.
    2.Identify the plate region to get scaling factors.
    3.For each other labeled region, calculate volume and store it in vol_dict.
    Is the main function that outputs a dictionary (vol_dict) with volumes for each labeled region in the image.

    """
    lowest        = np.max(img)
    vol_dict      = {}
    len_per_pix   = .0
    depth_per_pix = .0
    with open(json_path, "r") as json_file:
        data = json.load(json_file)
        for shape in data["shapes"]:
            if (shape["label"]=="plate"):
                len_per_pix, depth_per_pix = get_scale(shape["points"], img, lowest)
                break
        for shape in data["shapes"]:
            label = shape["label"]
            if (label == "plate"):
                continue
            points = shape["points"]
            volume = cal_volume(points, img, len_per_pix, depth_per_pix, lowest)
            if (label in vol_dict):
                vol_dict[label] += volume
            else:
                vol_dict[label] = volume
    return vol_dict
img = cv2.imread("out.png", 0)
print(get_volume(img, "test.json"))
