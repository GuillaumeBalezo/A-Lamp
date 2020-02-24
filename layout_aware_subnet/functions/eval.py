# Evaluation functions on the MSO dataset
#
# Jianming Zhang, Stan Sclaroff, Zhe Lin, Xiaohui Shen,
# Brian Price and Radomír Mech. "Unconstrained Salient
# Object Detection via Proposal Subset Optimization."
# CVPR, 2016.
# Code written by Guillaume Balezo, 2020

import numpy as np
from functions.utils import get_iou
import h5py

def calc_ap(rec, prec, interval = None):
  """ Calculating average precision using 11 point averaging.
    We first linearly interpolate the PR curves, which turns out
    to be more robust to the number of points sampled on the PR cureve.
        Args:
            rec: recall.
            prec: precision.
            interval: intervall.
        Returns: ap: average precision.
  """
  if not interval:
    interval = np.arange(0,1,0.1)

  # linear interpolation
  ii = np.argsort(rec) # axis ?
  rec = np.take(rec, ii)
  prec = np.take(prec, ii)
  rec, ii = np.unique(rec, return_index = True)
  prec = np.take(prec, ii)
  Rq = np.arange(0, 1, 0.01)
  Pq = np.interp(Rq, rec, prec)
  Pq = np.nan_to_num(Pq, nan = 0)
  prec = np.copy(Pq)
  rec = np.copy(Rq)
  ap = 0
  for t in interval:
    p = prec[rec >= t].max(0)
    if p.size == 0:
      p = 0
    ap = ap + p / len(interval)
  return ap

def evaluate_bbox(imgList, res):
  """ Function for evaluation. It outputs the number of hits,
    prediction and ground truths for each image.
        Args:
            imgList: .
            res: Token of the record.
        Returns: TP: True positive.
                 NPred: .
                 NGT: .
  """
  TTP = np.zeros((len(res), len(imgList)))
  NPred = np.zeros((len(res), len(imgList)))
  NGT = np.zeros(len(imgList))
  for i in range(len(res)):
    pred_num = np.zeros(len(imgList))
    TP = np.zeros(len(imgList))
    for j in range(len(imgList)):
      anno = imgList[j]['anno']
      if len(anno.shape) == 2:
        NGT[j] = anno.shape[1]
      else:
        NGT[j] = 0
      pred_num[j] = res[i][j].shape[1]
      bboxes = get_gt_hit_boxes(res[i][j], anno, 0.5).reshape(4, -1)
      if bboxes.size == 0:
        TP[j] = 0
      else:
        TP[j] = bboxes.shape[1]
    TTP[i, :] = TP
    NPred[i, :] = pred_num

  TP = np.copy(TTP)
  return TP, NPred, NGT

def get_gt_hit_boxes(bboxes, anno, thresh):
  """ This function returns the correct detection windows given
    the ground truth.
        Args:
            bboxes: bounding boxes.
            anno: annotation of the bboxes.
            thresh: thresold.
        Returns: res: .
  """

  res = np.array([])
  if anno.size == 0 or bboxes.size == 0:
    return res
  for i in range(anno.shape[1]):
    iou = get_iou(bboxes.T, anno[:, i])
    idx = iou.argmax()
    score = iou[idx]
    if score > thresh:
      res = np.concatenate((res, bboxes[:, idx]))
      bboxes = np.delete(bboxes, idx, axis = 1)
      if bboxes.size == 0:
        break
  return res

def read_props(filename):
  """ Load proposal prediction to avoid computational time
    in benchmark_MSO.py
        Args:
            filename: name of the pickle file.
        Returns: props: proposals.
  """
  return pickle.load(open(filename, "rb" ))

def save_props(props, filename):
  """ Save proposal prediction to avoid computational time
    in benchmark_MSO.py
        Args:
            filename: name of the pickle file.
            props: proposals.
  """
  pickle.dump(props, open( filename, "wb" ))

def load_MSO_dataset(filepath = 'imgIdx.mat'):
  """ Load MSO dataset. In python, it requires a special trick.
        Args:
            filepath: path of the file imgIdx.mat.
  """
  with h5py.File(filepath, 'r') as f:
    ref_names = np.array(f['imgIdx']['name'])
    ref_annos = np.array(f['imgIdx']['anno'])
    nb_images = ref_names.shape[0]
    imgIdx = []
    for i in range(nb_images):
      imgIdx.append({})
      name_encoded = np.array(f[ref_names[i, 0]])
      name = ''
      for code in name_encoded:
        name += chr(code[0])
      imgIdx[i]['name'] = name
      anno = np.array(f[ref_annos[i, 0]])
      if anno.size == 2:
        anno = np.array([])
      imgIdx[i]['anno'] = anno
  return imgIdx
