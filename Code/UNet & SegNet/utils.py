import tensorflow as tf
import tensorlayer as tl
from tensorlayer.prepro import *
import scipy
import numpy as np
import skimage
from skimage.transform import rotate
from skimage.io import imread
import random
    

def upside_down(im):
    degree = 180
    return rotate(im, degree)

def tissue_augment(im_tumor,is_mask=False):
    if not is_mask:
        choice=random.randint(0,13)
        im_free=imread('./bibu/'+str(choice)+'.png')
        im_free = im_free.reshape((im_free.shape[0], im_free.shape[1], 1))
        im_free = im_free/np.max(im_free)
        im_tumor = im_tumor/np.max(im_tumor)
        img3 = (im_free+im_tumor)/2
        return img3
    return im_tumor

def rand_rotate(im):
    degree=90 
    return rotate(im,degree)

def flip(im):
    return np.fliplr(im)


def data_augment(im,is_mask=False):
    
    return rand_rotate(im),flip(im),upside_down(im)

def get_imgs_fn(file_name, path):
    
    return scipy.misc.imread(path + file_name, mode='RGB')

def crop_sub_imgs_fn(x, is_random=True):
    x = crop(x, wrg=224, hrg=224, is_random=is_random)
    x = x / (255. / 2.)
    x = x - 1.
    
    
    return x
    
   
def downsample_fn(x):
   
    
    x = imresize(x, size=[56, 56], interp='bicubic', mode=None)
    x = x / (255. / 2.)
    x = x - 1.
    
    
    
    return x
    
    

def computePSNR(HR_img_path, to_compared_img_path):
    
    hr_img = scipy.misc.imread(HR_img_path, mode='L')
    to_be_compared_img = scipy.misc.imread(to_compared_img_path, mode='L')
    
    return skimage.measure.compare_psnr(hr_img, to_be_compared_img)


def computeSSIM(HR_img_path, to_compared_img_path):
    
    hr_img = scipy.misc.imread(HR_img_path, mode='L')
    to_be_compared_img = scipy.misc.imread(to_compared_img_path, mode='L')

    return skimage.measure.compare_ssim(hr_img, to_be_compared_img)


def computeSSIM_WithDataRange(HR_img_path, to_compared_img_path):
    
    hr_img = scipy.misc.imread(HR_img_path, mode='L')
    to_be_compared_img = scipy.misc.imread(to_compared_img_path, mode='L')

    return skimage.measure.compare_ssim(hr_img, to_be_compared_img, data_range=to_be_compared_img.max() - to_be_compared_img.min())
