from FCN_CrackAnalysis import CrackAnalyse
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
import cv2
import numpy as np
# import distancemap as dm
from skimage.feature import peak_local_max
from scipy.ndimage.measurements import center_of_mass, label


analyser = CrackAnalyse('C:/Users/swava/RSU Crack Segmentation/Crack Width Samples/CrackWidthDetection/Width/1.png')
crack_skeleton = analyser.get_skeleton()
crack_lenth = analyser.get_crack_length()
crack_max_width = analyser.get_crack_max_width()
crack_mean_width = analyser.get_crack_mean_width()


crack_prediction = analyser.get_prediction()
crack_segmentation = analyser.get_segmentation()
crack_lens = analyser.get_crack_lens()
crack_wids = analyser.get_crack_wids()

crack_ratio = analyser.get_ratio()

crack_coordinates = analyser.get_coordinates()


plt.figure(figsize=(10,5))


plt.subplot(1,3,1)
plt.imshow(crack_prediction, cmap='magma') #cmap='magma'
plt.title('crack_prediction')
plt.subplot(1,3,2)
plt.grid(False)
plt.imshow(crack_segmentation, cmap='magma') #cmap='magma'
plt.title('crack_segmentation')
plt.subplot(1,3,3)
plt.grid(False)
plt.imshow(crack_skeleton,cmap='magma')
plt.title('crack_skeleton')

plt.suptitle('Crack total length: '+str(round(crack_lenth,4))+' px' +'\nCrack max width: '+str(round(crack_max_width,4))+' px' +'\nCrack mean width: '+str(round(crack_mean_width,4))+' px')

plt.show()
plt.close()

plt.figure(figsize=(10,5))

plt.imshow(crack_prediction, cmap='magma') #cmap='magma'
plt.text(10, -60, 'Crack max width: '+str(round(crack_max_width,4))+' px', bbox=dict(fill=False, edgecolor='white', linewidth=2))
# plt.text(10, -100, 'Crack mean width: '+str(round(crack_mean_width,4))+' px', bbox=dict(fill=False, edgecolor='white', linewidth=2))
plt.text(10, -220, 'Crack total length: '+str(round(crack_lenth,4))+' px', bbox=dict(fill=False, edgecolor='white', linewidth=2))
plt.plot(crack_coordinates[:, 1], crack_coordinates[:, 0], 'r.')

plt.show()
plt.close()


