# File for setting the parameters of the model
#
# Jianming Zhang, Stan Sclaroff, Zhe Lin, Xiaohui Shen,
# Brian Price and Radomír Mech. "Unconstrained Salient
# Object Detection via Proposal Subset Optimization."
# CVPR, 2016.
# Code written by Guillaume Balezo, 2020

import numpy as np

def get_param(modelName, weights_path, center_path):
  param = {}
  param['modelName'] = modelName
  # See the paper for the meaning of the following three parameters
  param['lambda'] = 0.075
  param['gamma'] = 1
  param['phi'] = np.log(0.3)
  # The maximum output number
  param['maxnum'] = 30
  # By default, we perturb the initialization of our optimization for better
  # local maxima
  param['perturb'] = True
  # The number of proposals used from the whole image
  param['masterImgPropN'] = 30
  # The number of proposals used from each sub-image
  param['subImgPropN'] = 10
  # The number of sub-images (rois)
  param['roiN'] = 5
  # This parameter is used for merging similar rois
  param['roiClusterCutoff'] = 0.3
  param['roiExpand'] = 1
  # 100 proposal centers
  param['center'] = np.load(center_path)
  # The following parameters are for the CNN model
  param['weightsFile'] = weights_path
  param['useGPU'] = True
  param['GPUID'] = 0
  param['width'] = 224
  param['height'] = 224
  param['batchSize'] = 8

  return param
