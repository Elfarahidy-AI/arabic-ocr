from skimage import measure, filters, color
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize
from commonfunctions import *


def plot_text_regions(image, bounding_boxes):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(image, cmap='gray')

    for bbox in bounding_boxes:
        minr, minc, maxr, maxc = bbox
        rect = plt.Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
    img_array = np.array(image)    
    plt.yticks(np.arange(0, img_array.shape[0], 20))  # Set y ticks from 0 to image height at intervals of 50    
    plt.xticks(np.arange(0, img_array.shape[1], 50))  # Set x ticks from 0 to image width at intervals of 50
    plt.grid(True)
    plt.title('Text Regions Identified')
    plt.show()

