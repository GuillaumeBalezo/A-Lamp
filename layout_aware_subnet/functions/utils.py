# Utils functions
#
# Jianming Zhang, Stan Sclaroff, Zhe Lin, Xiaohui Shen,
# Brian Price and Radomír Mech. "Unconstrained Salient
# Object Detection via Proposal Subset Optimization."
# CVPR, 2016.
# Code written by Guillaume Balezo, 2020

import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img

def get_iou_float(res, gt):
  xmin = np.maximum(res[:, 0], gt[0])
  ymin = np.maximum(res[:, 1], gt[1])
  xmax = np.minimum(res[:, 2], gt[2])
  ymax = np.minimum(res[:, 3], gt[3])
  I = np.maximum(xmax - xmin, 0) * np.maximum(ymax - ymin, 0)
  U = (res[:, 2] - res[:, 0]) * (res[:, 3] - res[:, 1]) + (gt[2] - gt[0]) * (gt[3] - gt[1])
  iou = I / (U - I)
  if iou.size == 1:
    iou = iou.item()
  return iou

def get_roi_bbox(B, ROI):
  """ Translate the coordinates back into the original image.
  """
  ratio = ROI[2:] - ROI[:2]
  B = B * np.tile(ratio, 2).reshape(-1, 1)
  B = B + np.tile(ROI[:2], 2).reshape(-1, 1)
  return B

def expand_roi(roi, imsz, margin):
  roi[: 2, :] = roi[: 2, :] - margin
  roi[2 :, :] = roi[2 :, :] + margin
  roi[: 2, :] = np.maximum(roi[: 2, :], 0)
  roi[2, :] = np.minimum(roi[2, :], imsz[1])
  roi[3, :] = np.minimum(roi[3, :], imsz[0])
  return roi

def get_iou(res, gt):
  xmin = np.maximum(res[:, 0], gt[0])
  ymin = np.maximum(res[:, 1], gt[1])
  xmax = np.minimum(res[:, 2], gt[2])
  ymax = np.minimum(res[:, 3], gt[3])

  I = np.maximum((xmax - xmin + 1),0) * np.maximum((ymax - ymin + 1), 0)
  U = (res[:, 2] - res[:, 0] + 1) * (res[:, 3] - res[:, 1] + 1) + (gt[2] - gt[0] + 1) * (gt[3] - gt[1] + 1)
  iou = I / (U - I)
  return iou

def get_max_inc_float(res, gt):
  """ Another way to define window similarity:
  Intersection over min area
  """
  xmin = np.maximum(res[:, 0], gt[0])
  ymin = np.maximum(res[:, 1], gt[1])
  xmax = np.minimum(res[:, 2], gt[2])
  ymax = np.minimum(res[:, 3], gt[3])
  I = np.maximum((xmax - xmin), 0) * np.maximum((ymax - ymin), 0)
  U1 = (res[:, 2] - res[:, 0]) * (res[:, 3] - res[:, 1])
  U2 = (gt[2] - gt[0]) * (gt[3] - gt[1])
  inc = I / np.minimum(U1, U2)
  return inc

def do_nms(bboxes, thresh):
  if bboxes.size == 0: #not sure
    return bboxes, None

  tmp = bboxes[:, 0].reshape(-1, 1)
  idx = 0
  for i in range(1, bboxes.shape[1]):
    bboxes_i = bboxes[:, i].reshape(4, -1)
    iou = np.array(get_iou_float(tmp.T, bboxes_i))
    if (iou < thresh).all():
      tmp = np.hstack((tmp, bboxes_i))
      idx = np.hstack((idx, i))

  bboxes = np.copy(tmp)
  return bboxes, idx

def do_mmr(bbox, score, lambda_):
  """ An implementation of the Maximum Marginal
    Relevance re-ranking method used in this
    paper:
        J. Carreira and C. Sminchisescu. CPMC:
        Automatic object segmentation using
        constrained parametric min-cuts. PAMI
        34(7):1312–1328, 2012.
  """

  bbox = np.hstack((bbox, score.reshape(-1, 1)))
  if bbox.size == 0:
    return P, S

  res = bbox[0].reshape(1, -1)
  bbox = np.delete(bbox, 0, axis = 0)
  while bbox.size != 0:

    ss = bbox[:, -1]
    for i in range(bbox.shape[0]):
      ss[i] = ss[i] - lambda_ * get_iou(res, bbox[i, :4]).max(0)
    iidx = np.argsort(-ss, axis = 0)
    ss = ss[iidx]
    bbox = bbox[iidx, :]
    res = np.append(res, np.hstack((bbox[0, :4], ss[0])).reshape(1, -1), axis = 0)
    bbox = np.delete(bbox, 0, axis = 0)


  P = res[:, :4].T
  S = res[:, -1]
  return P, S

def imread_rgb(fileNames):
  I_batch = []
  for i in range(len(fileNames)):
    I_batch.append(np.array(load_img(fileNames[i])))
  return I_batch

def display_preds(fn, res, color = (255, 0, 0)):
  img = cv2.imread(fn)
  for i in range(res.shape[1]):
    rect = res[:, i].astype(int)
    img = cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), color, thickness = 5)
  cv2.imshow(img)


def init_model(param):
  """ Initialize the CNN model
    Input:
        weights_path: path of h5 file with the weights
        center_path: path of npy file with the center bounding boxes
    Output:
        model: SOD cnn model
  """
  vgg16 = tf.keras.applications.vgg16.VGG16(include_top=False, weights=None, input_shape=(224, 224, 3))
  flat = tf.keras.layers.Flatten()(vgg16.output)
  fc6 = tf.keras.layers.Dense(4096, activation='relu')(flat)
  fc7 = tf.keras.layers.Dense(4096, activation='relu')(fc6)
  fc8 = tf.keras.layers.Dense(100, activation='sigmoid')(fc7)
  model = tf.keras.models.Model(inputs=[vgg16.input], outputs=[fc8])
  model.load_weights(param['weightsFile'])
  model.trainable = False
  return model
