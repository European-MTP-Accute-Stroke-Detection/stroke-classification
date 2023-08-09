"""
Author: Eunmi Joo

Data Preprocessing Functions

- transform_to_hu
- window_image
- get_first_of_dicom_field_as_int
- get_windowing
- view_images
- convert_pixelarray_to_rgb
- mask_image
- align_image
- center_image
- scale_image
- check_mask_size
- data_preprocess

example code:

dicom = pydicom.read_file("/ceph/inestp02/stroke_classifier/data/Ischemic/data/0019983/000001.dcm")
dicom_preprocess, img_preprocess = data_preprocess(dicom)
plt.imshow(img_preprocess)
plt.show()
dicom_preprocess.save_as('./dicom_preprocess.dcm')

"""

import pydicom, numpy as np
import matplotlib.pylab as plt
import os
import pickle
import scipy.ndimage as ndi
import math
import cv2
from skimage import morphology
import pydicom as dicom

def transform_to_hu(img, intercept, slope):
    """Transform raw pixel values in CT scans to the Hounsfield Units (HU)
        HU: A standardized scale used to measure tissue density.

    Args:
        img (numpy array): Waw pixel values of CT scan
        intercept (int): Linear attenuation coefficients of the tissues.
        slope (int): Linear attenuation coefficients of the tissues.

    Returns:
        numpy array: Converted HU values from raw pixel values of img
    """
    hu_image = img * slope + intercept

    return hu_image

def window_image(img, window_center, window_width, intercept, slope, rescale=True):
    """Apply windowing to enhance specific regions of interest (brain, bone, subdural)

    Args:
        img (numpy array): Raw pixel values of CT scan
        window_center (int): Midpoint or central value of the selected range of pixel values
        window_width (int): Determine the range of pixel values that will be displayed.
        intercept (int): Linear attenuation coefficient (intercept) of the tissues.
        slope (int): Linear attenuation coefficient (slope) of the tissues.
        rescale (bool, optional): Extra rescaling to 0-1. Defaults to True.

    Returns:
        numpy array: windowed image
    """
    img = transform_to_hu(img, intercept, slope)
    is_zero = img == 0
    img_min = window_center - window_width//2
    img_max = window_center + window_width//2
    img[img<img_min] = img_min
    img[img>img_max] = img_max

    if rescale:
        # Extra rescaling to 0-1, not in the original notebook
        img = (img - img_min) / (img_max - img_min)
    
    img[is_zero] = 0
    
    return img
    
def get_first_of_dicom_field_as_int(x):
    #get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)
    if type(x) == pydicom.multival.MultiValue:
        return int(x[0])
    else:
        return int(x)

def get_windowing(data):
    """Get default windowing value

    Args:
        data (dicom format): Input image for getting windowing values

    Returns:
        int, int, int, int: window_center , window_width, intercept, slope
    """
    dicom_fields = [data[('0028','1050')].value, #window center
                    data[('0028','1051')].value, #window width
                    data[('0028','1052')].value, #intercept
                    data[('0028','1053')].value] #slope
    return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]

    
    
def view_images(images, title = '', aug = None):
    width = 5
    height = 2
    fig, axs = plt.subplots(height, width, figsize=(15,5))
    
    for im in range(0, height * width):
        data = pydicom.read_file(os.path.join(TRAIN_IMG_PATH,'ID_'+images[im]+ '.dcm'))
        image = data.pixel_array
        window_center , window_width, intercept, slope = get_windowing(data)
        image_windowed = window_image(image, window_center, window_width, intercept, slope)


        i = im // width
        j = im % width
        axs[i,j].imshow(image_windowed, cmap=plt.cm.bone) 
        axs[i,j].axis('off')
        
    plt.suptitle(title)
    plt.show()
    
def convert_pixelarray_to_rgb(image1, image2, image3):
    """Combine image1, image2, image3 into 3-channels image

    Args:
        image1 (numpy array): Windowed image (Brain)
        image2 (numpy array): Windowed image (Subdural)
        image3 (numpy array): Windowed image (Bone)

    Returns:
        numpy array: Combined 3-channels image
    """

    image = np.zeros((image1.shape[0], image1.shape[1], 3))
    
    image[:,:,0] = image1
    image[:,:,1] = image2
    image[:,:,2] = image3

    return image

def mask_image(brain_image, dilation = 12):
    """Find brain mask

    Args:
        brain_image (numpy array): Windowed image (Brain)
        dilation (int, optional): Dilation distance. Defaults to 12.

    Returns:
        numpy array: Mask image for brain_image
    """
    
    segmentation = morphology.dilation(brain_image, np.ones((dilation, dilation)))
    labels, label_nb = ndi.label(segmentation)

    label_count = np.bincount(labels.ravel().astype(np.int))
    label_count[0] = 0
    mask = labels == label_count.argmax()
    mask = morphology.dilation(mask, np.ones((3, 3)))
    mask = ndi.morphology.binary_fill_holes(mask)
    mask = morphology.dilation(mask, np.ones((3, 3)))
        
    return mask

error_contour = []

def align_image(image):
    """Align brain image symmetrically

    Args:
        image (numpy array): Mask image of brain image

    Returns:
        numpy array: Aligned brain image
    """

    img=np.uint8(image)
    contours, hier =cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # find the biggest contour (c) by the area
    c = max(contours, key = cv2.contourArea)

    try:
        (x,y),(MA,ma),angle = cv2.fitEllipse(c)
    except:
        error_contour.append(c)
        return img

    cv2.ellipse(img, ((x,y), (MA,ma), angle), color=(0, 255, 0), thickness=2)

    rmajor = max(MA,ma)/2
    if angle > 90:
        angle -= 90
    else:
        angle += 96
    xtop = x + math.cos(math.radians(angle))*rmajor
    ytop = y + math.sin(math.radians(angle))*rmajor
    xbot = x + math.cos(math.radians(angle+180))*rmajor
    ybot = y + math.sin(math.radians(angle+180))*rmajor
    cv2.line(img, (int(xtop),int(ytop)), (int(xbot),int(ybot)), (0, 255, 0), 3)

    M = cv2.getRotationMatrix2D((x, y), angle-90, 1)  #transformation matrix

    img = cv2.warpAffine(image, M, (img.shape[1], img.shape[0]), cv2.INTER_CUBIC)
    return img

def center_image(image, com, dim):
    # image centering

    if dim == 3:
        height, width, _ = image.shape
        shift = (height/2-com[0], width/2-com[1], 0)
    else:
        height, width = image.shape
        shift = (height/2-com[0], width/2-com[1])
    res_image = ndi.shift(image, shift)
    return res_image

def scale_image(image, mask):
    """Scale the brain image to its full size within the image.

    Args:
        image (numpy array): Input brain image
        mask (numpy array): Mask image of brain image

    Returns:
        numpy array: Scaled brain image
    """

    height, width = image.shape
    coords = np.array(np.nonzero(mask))
    top_left = np.min(coords, axis=1)
    bottom_right = np.max(coords, axis=1)
    
    cropped_image = image[top_left[0]:bottom_right[0],
                            top_left[1]:bottom_right[1]]
    hc, wc = cropped_image.shape
    
    size = np.max([hc, wc])

    top_left = int((size - hc) / 2)
    bottom_right = int((size - wc) / 2)

    cropped_img_sqr = np.zeros((size, size))
    cropped_img_sqr[top_left:top_left+hc, bottom_right:bottom_right+wc] = cropped_image
    cropped_img_sqr = cv2.resize(cropped_img_sqr, (height,width), interpolation=cv2.INTER_LINEAR)
    
    return cropped_img_sqr

def check_mask_size(mask, image, threshold=[0.1, 0.9]):
    """
    Check the brain covers less than a threshold portion of the original image.
    Args:
        mask (numpy array): Binary mask of the image.
        images (numpy array): Input image for checking brain size.
        threshold (float): Minimum portion of the image that the mask must cover.
    Returns:
        Boolean. True, if the mask is smaller than the threshold.
                 False, otherwise.
    """
    min_threshold = threshold[0]
    max_threshold = threshold[1]
    is_small_mask = False
    is_no_mask = False
    image_size = np.prod(image.shape)
    mask_size = np.count_nonzero(mask)
    mask_coverage = mask_size / image_size
    if mask_coverage < min_threshold:
        is_small_mask = True
    if mask_coverage > max_threshold:
        is_no_mask = True
    
    is_abnormal = is_small_mask or is_no_mask
    return is_abnormal

def data_preprocess(data):
    """
    Apply data preprocessing steps:
        Image Denoising
        Image Alignment
        Image Scaling
        Image Centering
        Image Windowing
        Image Resizing
    
    If brain size is too small only proceed with:
        Image Centering
        Image Windowing
        Image Resizing
    Args:
        Dicom: input dicom image for preprocessing
    Returns:
        Dicom: preprocessed image before windowing and resizing
        Numpy Array: preprocessed image (224, 224, 3)
    """
    
    img = data.pixel_array
    window_center , window_width, intercept, slope = get_windowing(data)
    
    dicom_preprocess = data # for display preprocessed image after prediction
 
    brain_image = window_image(img, 40, 80, intercept, slope) #bone windowing
    image_mask = mask_image(brain_image)
    
    # exclude small brain image - too small for detecting stroke
    # need additional process for handling a label and training
    is_abnormal = check_mask_size(image_mask, img)
    
    if not is_abnormal:
        img = image_mask * img
        img = align_image(img)
        # img = scale_image(img, image_mask)
        
    com = np.average(np.nonzero(brain_image), axis=1)
    img = center_image(img, com, img.ndim) # no needs for centering, applied in scaling step
    
    # save preprocessed dicom image for displaying after prediction (without windowing, resizing)
    img_uint16 = img.astype(np.uint16)
    dicom_preprocess.PixelData = img_uint16.tobytes()
    
    # windowing: brain, subdural, bone
    img2 = window_image(img, 40, 80, intercept, slope)
    img3 = window_image(img, 80, 200, intercept, slope)
    img4 = window_image(img, 600, 2800, intercept, slope)

    # combine 3 images into RGB channels
    img = convert_pixelarray_to_rgb(img2, img3, img4)

    img = cv2.resize(img, (224,224), interpolation=cv2.INTER_LINEAR)
    img_preprocess = img 
    
    return dicom_preprocess, img_preprocess, is_abnormal

