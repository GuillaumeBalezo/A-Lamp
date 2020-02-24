# Subset optimization functions
#
# Jianming Zhang, Stan Sclaroff, Zhe Lin, Xiaohui Shen,
# Brian Price and Radomír Mech. "Unconstrained Salient
# Object Detection via Proposal Subset Optimization."
# CVPR, 2016.
# Code written by Guillaume Balezo, 2020

import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img
from functions.utils import do_nms, get_max_inc_float, get_iou_float, expand_roi
from functions.get_Proposals import get_proposals

def prop_opt(bboxes, bboxscore, param):
  """ The main function for the subset optimization.
      Args:
          bboxes: Bounding boxes.
          bboxscore: Scores of the bboxes.
          param: parameters of the model.
      Returns: res: Bounding boxes after the optimization.
               stat: _:
  """

    # for the special case when lambda == 0
  if param['lambda'] == 0:
    stat = {}
    res = bboxes.copy()
    stat['O'] = np.arange(bboxes.shape[1]).reshape(-1, 1)
    return res, stat

  stat = do_map_forward(bboxes, bboxscore.astype(float), param)
  if stat['O'].size > 1:
    stat = do_map_backward(bboxes, bboxscore.astype(float), param, stat)

  # We use the second output to intialized the optimization again
  if param['perturb'] and len(stat['O']) > 1:
    # use the second output to initialize the forward pass
    statTmp = do_map_eval(bboxes, bboxscore.astype(float), param, stat['O'][1], stat['W'], stat['BGp'])
    statTmp = do_map_forward(bboxes, bboxscore.astype(float), param, statTmp)
    if statTmp['f'] > stat['f']:
      stat = statTmp.copy()
  res = np.take(bboxes, stat['O'].flatten(), axis = 1).copy()
  return res, stat


def do_map_forward(B, S, param, stat = None):
  if B.size == 0:
    print('Empty proposal set')
    stat = {}
    return stat

  nB = B.shape[1]
  if not stat:
    # initialization
    stat = {}
    stat['W'] = np.array([])
    stat['Xp'] = np.array([]) # optimal w_{ij} given the output set
    stat['X'] = np.zeros((nB, 1)) # assignment
    # construct W
    stat['W'], stat['Xp'] = get_w(B, S, param)
    stat['BGp'] = stat['Xp'].copy()
    stat['nms'] = np.zeros((B.shape[1], 1))
    stat['f'] = stat['Xp'].sum()
    stat['O'] = np.array([], dtype=int)

    ## loop
    while len(stat['O']) < min(param['maxnum'], nB):
      V = np.maximum(stat['W'] - stat['Xp'].reshape(-1, 1), 0)
      scores = V.sum(axis = 0) + stat['nms'].flatten().T
      vote = np.argmax(scores)
      score = scores[vote]
      if score == 0: # no supporters
        break
      tmpf = stat['f'] + score + param['phi']

      if (tmpf > stat['f']):
        mask = V[:, vote] > 0
        stat['X'][mask] = vote
        stat['O'] = np.append(stat['O'], vote).reshape(-1, 1)
        stat['Xp'][mask] = stat['W'][mask, vote]
        stat['f'] = tmpf
        stat['nms'] = stat['nms'] + param['gamma'] * get_nms_penalty(B, B[:, vote]).reshape(-1, 1)
      else:
        break
  return stat

def do_map_backward(B, S, param, stat):
  while stat['O'].size != 0:
    flag = False
    bestStat = stat.copy()
    for i in range(len(stat['O'])):
      O = stat['O'].copy()
      O = np.delete(O, i)
      statTmp = do_map_eval(B, S, param, O, stat['W'], stat['BGp'])
      if statTmp['f'] > bestStat['f']:
        flag = True
        bestStat = statTmp.copy()
    stat = bestStat.copy()
    if not flag:
      break

  return stat

def do_map_eval(B, S, param, O, W = None, BGp = None):
  """ This function evaluate the target function
  given a output window set.
    Args:
        B: .
        S: .
        param: .
        O: .
        W: .
        BGp: .

    Returns: statTmp: .
  """
  statTmp = {}
  statTmp['W'] = np.array([])
  statTmp['Xp'] = np.array([]) #optimal w_{ij} given the output set

  statTmp['X'] = np.zeros((B.shape[1], 1)) # assignment
  if type(W).__name__ == 'NoneType' or type(BGp).__name__ == 'NoneType':
    # construct W
    statTmp['W'], statTmp['BGp'] = get_w(B, S, param)
  else:
    statTmp['W'] = W.copy()
    statTmp['BGp'] = BGp.copy()

  statTmp['nms'] = np.zeros((B.shape[1],1))
  statTmp['O'] = O.copy()
  statTmp['f'] = O.size * param['phi']
  for i in range(O.size):
    statTmp['f'] = statTmp['f'] + statTmp['nms'][O[i], 0]
    statTmp['nms'] = statTmp['nms'] + param['gamma'] * get_nms_penalty(B, B[:, O[i]]).reshape(-1, 1)

  if O.size == 0:
    statTmp['f'] = statTmp['f'] + np.sum(BGp)
    return statTmp

  tmp_val = statTmp['W'][:, O]
  ass = tmp_val.argmax(axis = 1)
  Xp = tmp_val.max(axis = 1, keepdims=True)
  mask = Xp > BGp.reshape(BGp.shape[0], -1)
  statTmp['X'][mask] = ass.reshape(-1, 1)[mask]
  statTmp['Xp'] = np.maximum(Xp, BGp)
  statTmp['f'] = statTmp['f'] + statTmp['Xp'].sum()
  return statTmp

def get_nms_penalty(B, b):
  p = -0.5 * (get_max_inc_float(B.T, b) + get_iou_float(B.T, b))
  return p

def get_w(B, S, param):
  """ Precompute all w_{ij}
      Args:
          B: bounding boxes.
          S: scores.
          param: parameters of the model.
      Returns: W: likelihood.
               Xp: likelihood of the optimal assignments given the current output set
  """

  P = np.zeros((B.shape[1], B.shape[1]))
  for i in range(B.shape[1]):
    P[i, :] = get_iou_float(B.T, B[:, i].T)
  P = P * S.reshape(-1, 1)
  P = np.hstack((P, param['lambda'] * np.ones((B.shape[1], 1))))
  np.seterr(divide='ignore')
  P = P / P.sum(axis = 1, keepdims=True)
  W = np.log(P)
  Xp = W[:, -1]
  W = W[:, : -1]
  return W, Xp

def refine_win(I, res, net, param):
  """ A heuristic way to further refine the output windows
    For each small output window, we run our method on the
    ROI again and extract the output that has the
    largest IOU with the orignal window for replacement.
    NMS is further applied to remove duplicate windows.
        Args:
            I: image.
            res: bounding boxes.
            net: CNN model
            param: parameters of the model.
        Returns: res: bounding boxes.
  """

  imsz = np.array([I.shape[0], I.shape[1]])
  param['lambda'] = 0.05
  for i in range(res.shape[1]):
    bb = res[:, i].reshape(4, -1)
    bbArea = (bb[2, 0] - bb[0, 0]) * (bb[3, 0] - bb[1, 0])
    # only refine small windows
    if bbArea < (0.125 * imsz[0] * imsz[1]):
      margin = (bb[2] - bb[0] + bb[3] - bb[1]) * 0.2
      bb = expand_roi(bb, imsz, margin).astype(int)[:, 0]
      Itmp = I[bb[1] : bb[3], bb[0] : bb[2], :]
      Ptmp, Stmp = get_proposals(Itmp, net, param)
      restmp, _ = prop_opt(Ptmp, Stmp, param)

      if restmp.size != 0:
        restmp = get_roi_bbox(restmp, bb)
        ii = np.array(get_iou_float(restmp.T, res[:, i])).argmax()
        res[:, i] = restmp[:, ii]

  res = do_nms(res, 0.5)
  return res
