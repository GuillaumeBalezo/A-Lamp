# Functions for generating the proposal set for optimization
#
# Jianming Zhang, Stan Sclaroff, Zhe Lin, Xiaohui Shen,
# Brian Price and Radomír Mech. "Unconstrained Salient
# Object Detection via Proposal Subset Optimization."
# CVPR, 2016.
# Code written by Guillaume Balezo, 2020

import numpy as np
from scipy import cluster
import cv2
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img
from functions.utils import get_iou_float, get_roi_bbox

def get_proposals(I_batch, net, param):
  """ Generate the proposal set for optimization.
        Args:
            I_batch: Images of the batch.
            net: CNN model.
            param: parameters of the model.
        Returns: P_batch: Prediction over the batch.
                 S_batch: Scores associated with the predictions:
  """
  Ip = prepare_image(I_batch, param)
  scores_batch = net.predict(Ip)
  P_batch = []
  S_batch = []
  for idx_batch in range(len(I_batch)):
    scores = scores_batch[idx_batch]
    I = I_batch[idx_batch]
    imsz = np.array([I.shape[0], I.shape[1]])
    top_idxs = np.argsort(-scores)
    scores = np.take(scores, top_idxs)
    BB = param['center'][:, top_idxs]
    P = BB[:, : param['masterImgPropN']].copy()
    S = scores[: param['masterImgPropN']].copy()
    # extract ROIs
    ROI = BB[:, : param['roiN']].copy()
    ROI = post_proc(ROI, imsz, param)
    ROI = cluster_boxes(ROI, param) # merge some ROI if needed
    # process ROIs
    Ip = crop_imgs_and_prepare(I.copy(), ROI, param)
    if Ip.size == 0:
      P_batch.append(P)
      S_batch.append(S)
      continue
    scores = net.predict(Ip)
    top_idxs = np.argsort(-scores, axis = 1)
    scores = np.take_along_axis(scores, top_idxs, axis = 1)
    for i in range(Ip.shape[0]):
      B = param['center'][:, top_idxs[i, : param['subImgPropN']]]
      roi = ROI[:, i] / np.tile(np.roll(imsz, 1), 2)
      B = get_roi_bbox(B.copy(), roi)
      P = np.hstack((P, B))
      S = np.hstack((S, scores[i, : param['subImgPropN']]))
    P_batch.append(P)
    S_batch.append(S)
  return P_batch, S_batch


def prepare_image(I_batch, param):
  """ Preprocess the images before the CNN predictions.
        Args:
            I_batch: Images of the batch.
            param: parameters of the model.
        Returns: Ip: Images of the batch preprocessed
  """
  Ip = np.zeros((len(I_batch), param['width'], param['height'], 3))
  for i in range(len(I_batch)):
    img = I_batch[i]
    img = preprocess_input(img, mode='caffe')
    Ip[i] = np.expand_dims(cv2.resize(img, (param['width'], param['height']), interpolation = cv2.INTER_LINEAR), axis = 0)
  return Ip

def cluster_boxes(BB, param):
  if BB.shape[1] < 2:
    ROI = np.copy(BB)
    return ROI

  D = []
  for i in range(BB.shape[1]):
    for j in range(i + 1, BB.shape[1]):
      D.append(1 - get_iou_float(BB[:, j].reshape(-1, 1).T, BB[:, i]))
  Z = cluster.hierarchy.linkage(D)
  T = cluster.hierarchy.fcluster(Z, param['roiClusterCutoff'], criterion = 'distance')
  ROI = np.vstack((BB[:2, T == 1].min(axis = 1, keepdims=True), BB[2:, T == 1].max(axis = 1, keepdims=True))) # initialisation for the for loop
  for i in range(2, T.max() + 1):
    ROI = np.hstack((ROI, np.vstack((BB[:2, T == i].min(axis = 1, keepdims=True), BB[2:, T == i].max(axis = 1, keepdims=True)))))
  return ROI

def post_proc(ROI, imsz, param):
  """ Post processing of the CNN predictions.
        Args:
            ROI: Region of interest.
            imsz: Image size.
            param: parameters of the model.
        Returns: ROI: Post-processed CNN predictions
  """
  # expand
  w = ROI[2] - ROI[0]
  h = ROI[3] - ROI[1]
  ROI[0] = ROI[0] - 0.5 * w * param['roiExpand']
  ROI[1] = ROI[1] - 0.5 * h * param['roiExpand']
  ROI[2] = ROI[0] + w * (1 + param['roiExpand'])
  ROI[3] = ROI[1] + h * (1 + param['roiExpand'])

  ROI = ROI * np.tile(np.roll(imsz, 1), 2).reshape(-1, 1)
  ROI[:2] = np.maximum(ROI[:2], 0)
  ROI[2] = np.minimum(ROI[2], imsz[1])
  ROI[3] = np.minimum(ROI[3], imsz[0])


  # removing
  area = (ROI[2] - ROI[0] + 1) * (ROI[3] - ROI[1] + 1)

  ROI = ROI[:, area < (0.9 * imsz[0] * imsz[1])]
  return ROI


def crop_imgs_and_prepare(img, roilist, param):
  """ Crop the image on the region of interest and preprocess the crops before the
    CNN network. The function is used in get_proposals, during the refinement step.
        Args:
            img: image.
            roilist: Regions of interest.
            param: parameters of the model.
        Returns: Ip: Preprocessed crops of the image
  """
  Ip = []
  if len(roilist.shape) == 1:
    roilist = roilist.reshape(-1, 1)
  for i in range(roilist.shape[1]):
    roi = roilist[:, i]
    img_cropped = img[int(roi[1]) : int(roi[3]) + 1, int(roi[0]) : int(roi[2]) + 1, :]
    img_cropped = preprocess_input(img_cropped, mode = 'caffe')
    Ip.append(cv2.resize(img_cropped, (param['height'], param['width']), interpolation = cv2.INTER_LINEAR))
  return np.array(Ip)
