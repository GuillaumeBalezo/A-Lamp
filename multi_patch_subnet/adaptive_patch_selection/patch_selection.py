#!/usr/bin/python

import sys

import os
import numpy as np
import pandas as pd
import cv2
import itertools
import pickle
import glob
import os
from scipy.optimize import minimize
from scipy.linalg import fractional_matrix_power, inv
from cv2 import cvtColor, GaussianBlur, Sobel, CV_64F, COLOR_BGR2HSV, COLOR_BGR2GRAY

#Import saliency map computation function
####################################################################
## Author:
##       Xiang Ruan
##       httpr://ruanxiang.net
##       ruanxiang@gmail.com
## License:
##       GPL 2.0
##       NOTE: the algorithm itself is patented by OMRON, co, Japan
##             my previous employer, so please do not use the algorithm in
##             any commerical product
## Version:
##       1.01
##
## ----------------------------------------------------------------
## A python implementation of manifold ranking saliency
## Usage:
##      import MR
##      import matplotlib.pyplot as plt
##      mr = MR.MR_saliency()
##      sal = mr.saliency(img)
##      plt.imshow(sal)
##      plt.show()
##
## Check paper.pdf for algorithm details 
## I leave all th parameters open to maniplating, however, you don't
## have to do it, default values work pretty well, unless you really
## know what you want to do to modify the parameters


import scipy as sp
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.data import camera
from scipy.linalg import pinv

cv_ver = int(cv2.__version__.split('.')[0])
_cv2_LOAD_IMAGE_COLOR = cv2.IMREAD_COLOR if cv_ver >= 3 else cv2.CV_LOAD_IMAGE_COLOR

class MR_saliency(object):
    """Python implementation of manifold ranking saliency"""
    weight_parameters = {'alpha':0.99,
                         'delta':0.1}
    superpixel_parameters = {'segs':200,
                             'compactness':10,
                             'max_iter':10,
                             'sigma':1,
                             'spacing':None,
                             'multichannel':True,
                             'convert2lab':None,
                             'enforce_connectivity':False,
                             'min_size_factor':0.5,
                             'max_size_factor':3,
                             'slic_zero':False}
    binary_thre = None

    def __init__(self, alpha = 0.99, delta = 0.1,
                 segs = 200, compactness = 10,
                 max_iter = 10, sigma = 1,
                 spacing = None, multichannel = True,
                 convert2lab = None, enforce_connectivity = False,
                 min_size_factor = 0.5, max_size_factor = 3,
                 slic_zero = False):
        self.weight_parameters['alpha'] = alpha
        self.weight_parameters['delta'] = delta
        self.superpixel_parameters['segs'] = segs
        self.superpixel_parameters['compactness'] = compactness
        self.superpixel_parameters['max_iter'] = max_iter
        self.superpixel_parameters['sigma'] = sigma
        self.superpixel_parameters['spacing'] = spacing
        self.superpixel_parameters['multichannel'] = multichannel
        self.superpixel_parameters['convert2lab'] = convert2lab
        self.superpixel_parameters['enforce_connectivity'] = enforce_connectivity
        self.superpixel_parameters['min_size_factor'] = min_size_factor
        self.superpixel_parameters['max_size_factor'] = max_size_factor
        self.superpixel_parameters['slic_zero'] = slic_zero

    def saliency(self,img):
        # read image
        img = self.__MR_readimg(img)
        # superpixel
        labels = self.__MR_superpixel(img)
        # affinity matrix
        aff = self.__MR_affinity_matrix(img,labels)
        # first round
        first_sal = self.__MR_first_stage_saliency(aff,labels)
        # second round
        fin_sal = self.__MR_final_saliency(first_sal,labels,aff)
        return self.__MR_fill_superpixel_with_saliency(labels,fin_sal)

    
    def __MR_superpixel(self,img):
        return slic(img,self.superpixel_parameters['segs'],
                    self.superpixel_parameters['compactness'],
                    self.superpixel_parameters['max_iter'],
                    self.superpixel_parameters['sigma'],
                    self.superpixel_parameters['spacing'],
                    self.superpixel_parameters['multichannel'],
                    self.superpixel_parameters['convert2lab'],
                    self.superpixel_parameters['enforce_connectivity'],
                    self.superpixel_parameters['min_size_factor'],
                    self.superpixel_parameters['max_size_factor'],
                    self.superpixel_parameters['slic_zero'])

    def __MR_superpixel_mean_vector(self,img,labels):
        s = sp.amax(labels)+1
        vec = sp.zeros((s,3)).astype(float)
        for i in range(s):
            mask = labels == i
            super_v = img[mask].astype(float)
            mean = sp.mean(super_v,0)
            vec[i] = mean
        return vec

    def __MR_affinity_matrix(self,img,labels):        
        W,D = self.__MR_W_D_matrix(img,labels)
        aff = pinv(D-self.weight_parameters['alpha']*W)
        aff[sp.eye(sp.amax(labels)+1).astype(bool)] = 0.0 # diagonal elements to 0
        return aff

    def __MR_saliency(self,aff,indictor):
        return sp.dot(aff,indictor)

    def __MR_W_D_matrix(self,img,labels):
        s = sp.amax(labels)+1
        vect = self.__MR_superpixel_mean_vector(img,labels)
        
        adj = self.__MR_get_adj_loop(labels)
        
        W = sp.spatial.distance.squareform(sp.spatial.distance.pdist(vect))
        
        W = sp.exp(-1*W / self.weight_parameters['delta'])
        W[adj.astype(np.bool)] = 0
        

        D = sp.zeros((s,s)).astype(float)
        for i in range(s):
            D[i, i] = sp.sum(W[i])

        return W,D

    def __MR_boundary_indictor(self,labels):
        s = sp.amax(labels)+1
        up_indictor = (sp.ones((s,1))).astype(float)
        right_indictor = (sp.ones((s,1))).astype(float)
        low_indictor = (sp.ones((s,1))).astype(float)
        left_indictor = (sp.ones((s,1))).astype(float)
    
        upper_ids = sp.unique(labels[0,:]).astype(int)
        right_ids = sp.unique(labels[:,labels.shape[1]-1]).astype(int)
        low_ids = sp.unique(labels[labels.shape[0]-1,:]).astype(int)
        left_ids = sp.unique(labels[:,0]).astype(int)

        up_indictor[upper_ids] = 0.0
        right_indictor[right_ids] = 0.0
        low_indictor[low_ids] = 0.0
        left_indictor[left_ids] = 0.0

        return up_indictor,right_indictor,low_indictor,left_indictor

    def __MR_second_stage_indictor(self,saliency_img_mask,labels):
        s = sp.amax(labels)+1
        # get ids from labels image
        ids = sp.unique(labels[saliency_img_mask]).astype(int)
        # indictor
        indictor = sp.zeros((s,1)).astype(float)
        indictor[ids] = 1.0
        return indictor

    def __MR_get_adj_loop(self, labels):
        s = sp.amax(labels) + 1
        adj = np.ones((s, s), np.bool)

        for i in range(labels.shape[0] - 1):
            for j in range(labels.shape[1] - 1):
                if labels[i, j] != labels[i+1, j]:
                    adj[labels[i, j],       labels[i+1, j]]              = False
                    adj[labels[i+1, j],   labels[i, j]]                  = False
                if labels[i, j] != labels[i, j + 1]:
                    adj[labels[i, j],       labels[i, j+1]]              = False
                    adj[labels[i, j+1],   labels[i, j]]                  = False
                if labels[i, j] != labels[i + 1, j + 1]:
                    adj[labels[i, j]        ,  labels[i+1, j+1]]       = False
                    adj[labels[i+1, j+1],  labels[i, j]]               = False
                if labels[i + 1, j] != labels[i, j + 1]:
                    adj[labels[i+1, j],   labels[i, j+1]]              = False
                    adj[labels[i, j+1],   labels[i+1, j]]              = False
        
        upper_ids = sp.unique(labels[0,:]).astype(int)
        right_ids = sp.unique(labels[:,labels.shape[1]-1]).astype(int)
        low_ids = sp.unique(labels[labels.shape[0]-1,:]).astype(int)
        left_ids = sp.unique(labels[:,0]).astype(int)
        
        bd = np.append(upper_ids, right_ids)
        bd = np.append(bd, low_ids)
        bd = sp.unique(np.append(bd, left_ids))
        
        for i in range(len(bd)):
            for j in range(i + 1, len(bd)):
                adj[bd[i], bd[j]] = False
                adj[bd[j], bd[i]] = False

        return adj
        
    def __MR_fill_superpixel_with_saliency(self,labels,saliency_score):
        sa_img = labels.copy().astype(float)
        for i in range(sp.amax(labels)+1):
            mask = labels == i
            sa_img[mask] = saliency_score[i]
        return cv2.normalize(sa_img,None,0,255,cv2.NORM_MINMAX)

    def __MR_first_stage_saliency(self,aff,labels):
        up,right,low,left = self.__MR_boundary_indictor(labels)
        up_sal = 1- self.__MR_saliency(aff,up)
        up_img = self.__MR_fill_superpixel_with_saliency(labels,up_sal)
    
        right_sal = 1-self.__MR_saliency(aff,right)
        right_img = self.__MR_fill_superpixel_with_saliency(labels,right_sal)

        low_sal = 1-self.__MR_saliency(aff,low)
        low_img = self.__MR_fill_superpixel_with_saliency(labels,low_sal)
    
        left_sal = 1-self.__MR_saliency(aff,left)
        left_img = self.__MR_fill_superpixel_with_saliency(labels,left_sal)

        return 1- up_img*right_img*low_img*left_img


    def __MR_final_saliency(self,integrated_sal, labels, aff):
        # get binary image
        if self.binary_thre == None:
            thre = sp.median(integrated_sal.astype(float))

        mask = integrated_sal > thre
        # get indicator
        ind = self.__MR_second_stage_indictor(mask,labels)
    
        return self.__MR_saliency(aff,ind)

    # read image
    def __MR_readimg(self,img):
        if isinstance(img,str): # a image path
            img = cv2.imread(img, _cv2_LOAD_IMAGE_COLOR)
        img = cv2.cvtColor(img,cv2.COLOR_RGB2LAB).astype(float)/255
        # Keeping the image in full size
        #h = 100
        #w = int(float(h)/float(img.shape[0])*float(img.shape[1]))
        #return cv2.resize(img,(w,h))
        return img



class SelectAdaptativePatch(object):
  def __init__(self, filename, image, S, patch_size=[112, 112]):
    self.filename = filename
    self.image = image
    self.patch_size = patch_size
    self.S = S

    self.E_x, self.E_y, self.H = self.preComputeF()

  def preComputeF(self):

    ## E_x, E_y edge maps (on x & y)
    gray = cvtColor(self.image, COLOR_BGR2GRAY)
    gray_clean = GaussianBlur(gray,(3,3),0)
    E_x = np.asarray(Sobel(gray_clean, CV_64F,1,0,ksize=5))
    E_y = np.asarray(Sobel(gray_clean, CV_64F,0,1,ksize=5))

    ## H chrominancy map ; we chose to use hue to score "colorfullness"
    #Hue formula from BGR following Frank Preucil, 1953
    H = np.arctan2(np.sqrt(3) * (self.image[:, :, 1] - self.image[:, :, 0]), 2 * self.image[:, :, 2] - self.image[:, :, 1] - self.image[:, :, 0])

    #pdb.set_trace()

    return E_x, E_y, H

  def wassertsteinDistance(self, sigma_i, sigma_j):
    # Following F. Pitie and A. Kokaram
    for i in range(sigma_i.shape[0]):
      if sigma_i[i][i] == 0: 
        sigma_i[i][i] += 0.1
      if sigma_j[i][i] == 0: 
        sigma_j[i][i] += 0.1

    sigma_i_sqrt = fractional_matrix_power(sigma_i, .5)
    sigma_i_sqrt_inv = np.linalg.inv(sigma_i_sqrt)
    sigma_intermediate = np.dot(sigma_i_sqrt, sigma_j)
    sigma_temp = fractional_matrix_power(np.nan_to_num(np.dot(sigma_intermediate, sigma_i_sqrt)), .5)

    #pdb.set_trace()

    return np.dot(np.dot(sigma_i_sqrt_inv, sigma_temp), sigma_i_sqrt_inv)

  def D_p(self, E_x_i, E_x_j, E_y_i, E_y_j, H_i, H_j):
    # Pattern Diversity
    ## Edge 
    sigma_e_x_i = np.var(E_x_i)
    sigma_e_x_j = np.var(E_x_i)

    sigma_e_y_i = np.var(E_y_i)
    sigma_e_y_j = np.var(E_y_i)

    ## Chrominance (Hue)
    sigma_h_i = np.var(H_i)
    sigma_h_j = np.var(H_j)

    # Earth's Mover Distance 
    sigma_i = np.diag([sigma_e_x_i, sigma_e_y_i, sigma_h_i])
    sigma_j = np.diag([sigma_e_x_j, sigma_e_y_j, sigma_h_j])

    emd = self.wassertsteinDistance(sigma_i, sigma_j)
    
    return np.trace(emd).mean()

  def initialize_five_centers_line(self):
    shape = self.image.shape[1], self.image.shape[0]
    safety = 1 

    if shape[1] > shape[0]:
      y0 = shape[1] // 2
      x_step = (shape[0] - 2*self.patch_size[0]) // 5
                                                                                
      x = [self.patch_size[0] + safety, y0,
            self.patch_size[0] + x_step, y0,
            self.patch_size[0] + 2*x_step, y0,
            self.patch_size[0] + 3*x_step, y0,
            self.patch_size[0] + 4*x_step - safety, y0]    

    else:
      x0 = shape[0] // 2
      y_step = (shape[1] - 2*self.patch_size[1]) // 5
                                                                                
      x = [x0, self.patch_size[1] + safety,
          x0, self.patch_size[1] + y_step, 
          x0, self.patch_size[1] + 2*y_step, 
          x0, self.patch_size[1] + 3*y_step, 
          x0, self.patch_size[1] + 4*y_step - safety] 
                  
    return np.array(x)

  def computeF(self, centers):
    #<0 : to minimize
    centers = centers.reshape(-1, 2).astype(int)

    #Init F
    F = 0

    ## To sum saliency, important : highly dependent on the combination method, here fitted for itertools.combinations
    first_loop = True
    number_loop = 0

    # Compute patches
    for center in itertools.combinations(centers, 2):
      #Careful : cv2 is switching axis so we have to switch centers coordinates accordingly

      #print(number_loop)
      E_x_i = self.E_x[(center[0][1] - self.patch_size[0]):(center[0][1] + self.patch_size[0] + 1), (center[0][0] - self.patch_size[1]):(center[0][0] + self.patch_size[1] + 1)]

      #2 * radius + 1 (center pixel)
      if (E_x_i.size == 0) or (E_x_i.shape[0] != self.patch_size[0] * 2 + 1) or (E_x_i.shape[1] != self.patch_size[1] * 2 + 1):
        #print("The patch i is stepping outside of the image.")
        return 1

      E_x_j = self.E_x[(center[1][1] - self.patch_size[0]):(center[1][1] + self.patch_size[0] + 1), (center[1][0] - self.patch_size[1]):(center[1][0] + self.patch_size[1] + 1)]

      if (E_x_j.size == 0) or (E_x_j.shape[0] != self.patch_size[0] * 2 + 1) or (E_x_j.shape[1] != self.patch_size[1] * 2 + 1):
        # Given that all of the maps are of the same dimension, only E_x on i and j is enough
        #print("The patch j is stepping outside of the image.")
        return 1

      E_y_i = self.E_y[(center[0][1] - self.patch_size[0]):(center[0][1] + self.patch_size[0] + 1), (center[0][0] - self.patch_size[1]):(center[0][0] + self.patch_size[1] + 1)]
      E_y_j = self.E_x[(center[1][1] - self.patch_size[0]):(center[1][1] + self.patch_size[0] + 1), (center[1][0] - self.patch_size[1]):(center[1][0] + self.patch_size[1] + 1)]

      H_i = self.H[(center[0][1] - self.patch_size[0]):(center[0][1] + self.patch_size[0] + 1), (center[0][0] - self.patch_size[1]):(center[0][0] + self.patch_size[1] + 1)]
      H_j = self.H[(center[1][1] - self.patch_size[0]):(center[1][1] + self.patch_size[0] + 1), (center[1][0] - self.patch_size[1]):(center[1][0] + self.patch_size[1] + 1)]

      # Summing saliency
      if first_loop:
        F += self.S[(center[0][0] - self.patch_size[0]):(center[0][0] + self.patch_size[0] + 1), (center[0][1] - self.patch_size[1]):(center[0][1] + self.patch_size[1] + 1)].mean()
        first_loop = False
      if number_loop < centers.shape[0] - 1:
        F += self.S[(center[1][0] - self.patch_size[0]):(center[1][0] + self.patch_size[0] + 1), (center[1][1] - self.patch_size[1]):(center[1][1] + self.patch_size[1] + 1)].mean()
        number_loop += 1

      # Pattern diversity (emd)
      F = F + self.D_p(E_x_i, E_x_j, E_y_i, E_y_j, H_i, H_j)
      
      # Euclidean distance between centers
      F = F + np.linalg.norm(center[0] - center[1]) / 2 

    return - F

  def predict(self):
    #check if image needs padding (if smaller than self.patch_size)
    if self.image.shape[0] < self.patch_size[0]:
      pad = (self.patch_size[0] - self.image.shape[0]) // 2 + 1 
      self.image = cv2.copyMakeBorder(self.image, pad, pad, 0, 0)

    if self.image.shape[0] < self.patch_size[0]:
      pad = (self.patch_size[0] - image.shape[0]) // 2 + 1 
      self.image = cv2.copyMakeBorder(image, 0, 0, pad, pad)

    #Setup
    x0 = self.initialize_five_centers_line()

    res = minimize(lambda x : self.computeF(x), \
                        x0, \
                        method='Nelder-Mead', \
                        options={'xatol': 10, 'maxiter': 200})
    
    centers = res.x.reshape(-1, 2).astype(int)
    bboxes = np.concatenate((centers[:, [0]] - self.patch_size[0], centers[:, [1]] - self.patch_size[1], centers[:, [0]] + self.patch_size[0], centers[:, [1]] + self.patch_size[1]), axis=1)

    return bboxes


def main(argv):
    """
        2 argumnents : input_dir, output_dir
    """
        
    #get inputs
    input_dir = argv[0]
    output_dir = argv[1]
    
    #get all file names
    filenames = glob.glob(input_dir + '/*')
    
    #Init list savers 
    filenames_temp = []
    bboxes_temp = []
    
    #Init MR_saliency
    mr = MR_saliency()
    
    i = 0
    
    for filename in filenames:
      #First image is no 1
      i+=1
      
      #load image
      image = cv2.imread(filename)
      
      #Compute saliency
      S = mr.saliency(image)
    
      patch_size = [112, 112]
    
      #check if image needs padding (if smaller than patch_size)
      if image.shape[0] < patch_size[0]:
        pad = (patch_size[0] - image.shape[0]) // 2 + 1 
        image = cv2.copyMakeBorder(image, 0, 0, pad, pad, cv2.BORDER_REFLECT)
    
      if image.shape[1] < patch_size[1]:
        pad = (patch_size[1] - image.shape[1]) // 2 + 1 
        image = cv2.copyMakeBorder(image, pad, pad, 0, 0, cv2.BORDER_REFLECT)
    
      #Run selection of patches
      #init
      patch_selection = SelectAdaptativePatch(filename, image, S)
      #predict
      bboxes = patch_selection.predict()
    
      #save result
      filenames_temp.append(filename)
      bboxes_temp.append(bboxes)
    
      #save in pickle periodically
      if (i % 1000 == 0):
        df = pd.DataFrame({'Filename':filenames_temp, 'BBoxes': bboxes_temp})
        filenames_temp = []
        bboxes_temp = []
    
        pickle_out = open(os.path.join(output_dir, str(i) + '_bboxes.pickle'), 'wb')
        pickle.dump(df, pickle_out)
        pickle_out.close()
        
    
    
    df = pd.DataFrame({'Filename':filenames_temp, 'BBoxes': bboxes_temp})
    pickle_out = open(os.path.join(output_dir, str(i) + '_bboxes.pickle'), 'wb')
    pickle.dump(df, pickle_out)
    pickle_out.close()
    

if __name__ == "__main__":
   main(sys.argv[1:])
