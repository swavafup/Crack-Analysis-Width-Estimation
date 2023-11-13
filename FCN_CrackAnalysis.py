from __future__ import division, print_function, absolute_import
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu
from skimage import morphology, feature, measure, exposure, filters, color
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage import data, img_as_float
from scipy.ndimage.measurements import center_of_mass, label
import cv2
import imageio as imageio
import plotly.graph_objects as go
import pandas as pd
from numpy import asarray 


BLUE = '#00b8e6'
DARKBLUE = '#00343f'
RED = '#ff5983'
DARKRED = '#7a023c'
YELLOW = '#ffe957'
DARKYELLOW = '#f29f3f'
GREEN = '#61ff69'
DARKGREEN = '#0b6e48'
GRAY = '#cccccc'

class CrackAnalyse(object):
    def __init__(self, predict_image_file):
        # load
        img = imageio.imread(predict_image_file, pilmode='L')
        img_size = img.size
        self.img = img

        # binary
        img_bnr = (img > 0).astype(np.uint8)

        # opening and closing
        img_bnr = ndi.morphology.binary_closing(img_bnr)
        img_bnr = ndi.morphology.binary_opening(img_bnr)

        self.img_bnr = img_bnr

        # segmentation
        img_labels, num_labels = ndi.label(img_bnr)
        # background label = 0
        labels = range(1, num_labels + 1)
        sizes = ndi.sum(img_bnr, img_labels, labels)

        # argsort according to size descend
        order = np.argsort(sizes)[::-1]
        labels = [labels[i] for i in order]
        # print(len (labels))
        
        total_number_of_labels = len (labels)

        img_sgt = img_labels / np.max(labels)
        self.img_sgt = img_sgt

        crack_lens = []
        crack_max_wids = []
        img_skl = np.zeros_like(self.img, dtype=np.float32)
        
        all_crack_image = []
        merged_crack_image = []

        crack_max_len_flag = 5000
        
        CrackMergedImage = []
        CrackAllMergedImage = []

        # skeletonize - median
        for label in labels:
            mask = img_labels == label
            # save the steps for analyse
            # imageio.imsave(str(label) + '.png', mask / (num_labels + 1))

            median_axis, median_dist = morphology.medial_axis(img_sgt, mask, return_distance=True)
            
            crack_len = np.sum(median_axis)
            crack_max_len = np.max(crack_len)
            crack_max_wid = np.max(median_dist)
            
            
            
            thresh = threshold_otsu(mask)
            binary = mask > thresh
            RGB = np.zeros((binary.shape[0],binary.shape[1],3), dtype=np.uint8)
            RGB[binary]  = [255,0,0]
            
            RGBO = np.zeros((binary.shape[0],binary.shape[1],3), dtype=np.uint8)
            RGBO[binary]  = [0, 0, 255]
            
            RGBG = np.zeros((binary.shape[0],binary.shape[1],3), dtype=np.uint8)
            RGBG[binary]  = [0,255,0]
            
            if crack_max_wid > 10:
                CrackImage = Image.fromarray(RGB)
            elif 5 <crack_max_wid <= 10:
                CrackImage = Image.fromarray(RGBO)

            elif 0 <crack_max_wid <= 5:
                CrackImage = Image.fromarray(RGBG)


            else:
                CrackImage = binary
    

            
            all_crack_image.append(CrackImage)
            

            img_mph = median_axis * median_dist

            crack_lens.append(crack_len)
        
            crack_max_wids.append(crack_max_wid)
            img_skl += img_mph
        
      

        cols = 3
        rows = 2
        print(rows)
        for i in range(0, len(all_crack_image), rows*cols):
            fig = plt.figure(figsize = (20,15))
            plt.axis('off')
            for j in range(0, rows*cols):
                fig.add_subplot(rows, cols, j+1)
                plt.axis('off')
                try:
                    print(crack_max_wids[j])
                    print("crack image: ", all_crack_image[i+j])
                    CrackMergedImage = asarray( all_crack_image[i+j]) 

                    print("crack array: ", CrackMergedImage)     

                    merged_crack_image= np.resize(merged_crack_image,CrackMergedImage.shape)
                    print("merged crack array: ", merged_crack_image)     


                    merged_crack_image = merged_crack_image + CrackMergedImage
                    print("Merged crack: ", merged_crack_image)        

                    if crack_max_wids[j] > 10:
                        plt.imshow(all_crack_image[i+j])
                        plt.axis('off')
                        plt.text(20, -354, 'Crack max width: '+str(round(crack_max_wids[j],4))+' px', bbox=dict(fill=False, edgecolor='black', linewidth=2))
                        plt.text(20, -124, 'Need to be repaired', bbox=dict(facecolor='#FF0000',boxstyle='round', fill=False, edgecolor='#FF0000', linewidth=2))

                    elif 5 <crack_max_wids[j] <= 10:
                        plt.imshow(all_crack_image[i+j])
                        plt.axis('off')
                        plt.text(20, -254, 'Crack max width: '+str(round(crack_max_wids[j],4))+' px', bbox=dict(fill=False, edgecolor='black', linewidth=2))
                        plt.text(20, -124, 'Medium crack', bbox=dict(facecolor='#0000FF',boxstyle='round',fill=False, edgecolor='#0000FF', linewidth=2))

                    elif 0 <crack_max_wids[j] <= 5:
                        plt.imshow(all_crack_image[i+j])
                        plt.axis('off')
                        plt.text(20, -254, 'Crack max width: '+str(round(crack_max_wids[j],4))+' px', bbox=dict(fill=False, edgecolor='black', linewidth=2))
                        plt.text(20, -124, 'Hairline crack', bbox=dict(facecolor='#00FF00',boxstyle='round', fill=False, edgecolor='#00FF00', linewidth=2))
                except:
                    plt.axis('off')
                    print("An exception occurred")
                
        # CrackAllMergedImage = np.array(merged_crack_image)
        # print("Merged crack array: ", CrackAllMergedImage)
        
        

        figMerge = plt.figure(figsize = (10,5))
        plt.axis('off')
        plt.imshow(merged_crack_image)
        plt.text(20, -124, 'Need to be repaired', bbox=dict(facecolor='#FF0000',boxstyle='round', fill=False, edgecolor='#FF0000', linewidth=2))
        plt.text(120, -24, 'Medium crack', bbox=dict(facecolor='#0000FF',boxstyle='round',fill=False, edgecolor='#0000FF', linewidth=2))
        plt.text(200, -24, 'Hairline crack', bbox=dict(facecolor='#00FF00',boxstyle='round', fill=False, edgecolor='#00FF00', linewidth=2))

        plt.show()
        '''
        # skeleton - zhand and suen
        for label in labels:
            mask = (img_labels == label).astype(np.uint8)
            crack_skl = morphology.skeletonize(mask)
            crack_len = np.sum(crack_skl)
            crack_width = np.sum(mask) / crack_len

            crack_lens.append(crack_len)
            crack_max_wids.append(crack_width)
            img_skl += crack_skl
        '''
        coordinates = peak_local_max(img_skl, min_distance=0)
        
        
        # print(coordinates)
        self.coordinates = coordinates
        
        self.img_mph = img_mph
        self.img_skl = img_skl
        self.crack_lens = np.array(crack_lens)
        self.crack_max_wids = np.array(crack_max_wids)
        self.ratio = np.sum(img_bnr) / img_size
    
    def get_coordinates(self):
        return self.coordinates

    def get_prediction(self):
        return self.img

    def get_segmentation(self):
        return self.img_sgt
    
    def get_skeleton(self):
        return ndi.grey_dilation(self.img_skl, size=2)

    def get_median(self):
        return ndi.grey_dilation(self.img_mph, size=2)

    def get_crack_lens(self):
        return self.crack_lens

    def get_crack_wids(self):
        return self.crack_max_wids

    def get_crack_length(self):
        return np.sum(self.crack_lens)

    def get_crack_max_width(self):
        return np.max(self.crack_max_wids)

    def get_crack_mean_width(self):
        return np.sum(self.img_bnr) / np.sum(self.crack_lens)

    def get_ratio(self):
        return self.ratio

    
class Edge_Detector(object):
    def __init__(self, original_image):
        img_gray = color.rgb2gray(original_image)
        self.img_gray = img_gray

    def get_edges(self, detector='sobel'):
        if detector == 'sobel':
            img = filters.sobel(self.img_gray)
        elif detector == 'canny1':
            img = feature.canny(self.img_gray, sigma=1)
        elif detector == 'canny3':
            img = feature.canny(self.img_gray, sigma=3)
        elif detector == 'scharr':
            img = filters.scharr(self.img_gray)
        elif detector == 'prewitt':
            img = filters.prewitt(self.img_gray)
        elif detector == 'roberts':
            img = filters.roberts(self.img_gray)
        return img

def Hilditch_skeleton(binary_image):
    size = binary_image.size
    skel = np.zeros(binary_image.shape, np.uint8)

    elem = np.array([[0, 1, 0],
                     [1, 1, 1],
                     [0, 1, 0]])

    image = binary_image.copy()
    for _ in range(10000):
        eroded = ndi.binary_erosion(image, elem)
        temp = ndi.binary_dilation(eroded, elem)
        temp = image - temp
        skel = np.bitwise_or(skel, temp)
        image = eroded.copy()

        zeros = size - np.sum(image > 0)
        if zeros == size:
            break

    return skel

