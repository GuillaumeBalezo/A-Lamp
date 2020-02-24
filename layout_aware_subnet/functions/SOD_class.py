# SOD class
#
# Jianming Zhang, Stan Sclaroff, Zhe Lin, Xiaohui Shen,
# Brian Price and Radomír Mech. "Unconstrained Salient
# Object Detection via Proposal Subset Optimization."
# CVPR, 2016.
# Code written by Guillaume Balezo, 2020

import numpy as np
from tqdm import tqdm
import time
from tensorflow.keras import backend as K
from functions.get_Param import get_param
from functions.utils import init_model, imread_rgb
from functions.get_Proposals import get_proposals
from functions.prop_Opt import prop_opt, refine_win


class SOD:

  def __init__(self, modelName = 'SOD_python', weights_path = '/content/drive/My Drive/Meero/weights/sod_cnn_weights.h5', center_path = '/content/drive/My Drive/Meero/layout_aware_subnet/SOD_python/center100.npy'):
    self.param = get_param(modelName, weights_path, center_path)
    K.clear_session()
    self.net = init_model(self.param)

  def split_batch(self, fns):
    m = len(fns)
    batches = []
    for i in range(0, m, self.param['batchSize']):
      batches.append(fns[i: i + self.param['batchSize']])
    if i + 10 < m:
      batches.append(fns[i:])
    return batches

  def predict(self, fns, refine = False, verbose = False):

    start = time.time()
    batches = self.split_batch(fns)
    res_all = []
    for batch in tqdm(batches):
      I_batch = imread_rgb(batch)
      P_batch, S_batch = get_proposals(I_batch, self.net, self.param)
      res_batch = []
      for idx_batch, (P, S) in enumerate(zip(P_batch, S_batch)):
        I = I_batch[idx_batch]
        imsz = [I.shape[0], I.shape[1]]
        res, _ = prop_opt(P, S, self.param)
        # scale bboxes to full size
        res = res * np.tile(np.roll(imsz, 1), 2).reshape(-1, 1)

        # optional window refining process
        if refine:
          I = np.expand_dims(I, axis = 0)
          res, _ = refine_win(I, res, self.net, self.param)

        res = res.astype(int)
        res_batch.append(res)

      res_all.extend(res_batch)

    if verbose:
      print('Time elapsed: {}'.format(round(time.time() - start, 2)))
      print('Time per images: {}'.format(round((time.time() - start)/len(fns), 2)))
    return res_all

  def predict_for_benchmarks(self, fns):
    batches = self.split_batch(fns)
    P_all = []
    S_all = []
    for batch in tqdm(batches):
      I_batch = imread_rgb(batch)
      P_batch, S_batch = get_proposals(I_batch, self.net, self.param)
      P_all.extend(P_batch)
      S_all.extend(S_batch)
    return P_all, S_all
