# Unconstrained Salient Object Detection

[![License](https://img.shields.io/packagist/l/doctrine/orm.svg)](LICENSE)

This is my implementation of the salient object detection method described in

> [Jianming Zhang, Stan Sclaroff, Zhe Lin, Xiaohui Shen, Brian Price and Radomír Mech. "Unconstrained Salient Object Detection via Proposal Subset Optimization." CVPR, 2016.](http://cs-people.bu.edu/jmzhang/sod.html)

The [original implementation](https://github.com/jimmie33/SOD) was in Matlab and Caffe. I converted it in Python and Tensorflow.


This method aims at producing a highly compact set of detection windows for salient objects in uncontrained images, which may or may not contain salient objects. Please cite the above paper if you find this work useful.

![Alt text](results/birds.jpg?raw=true "A prediction from SOD")

## Prerequisites
1. Linux
2. Python 3 
3. Tensorflow 2

## Quick Start
1. Unzip the files to a local folder.
2. Download the [weights](https://drive.google.com/open?id=1YJ82f34inEwJXGQtZYRtgw1Sxd7KVsSi)
3. Donwload [the MSO dataset](http://cs-people.bu.edu/jmzhang/sos.html)
4. Run **demo.py**.

**You can also run the jupyter notebook. If you don't want to install anything, you can execute the notebook with Google Colab.**
 
## Evaluation
You can reproduce the result on [the MSO dataset](http://cs-people.bu.edu/jmzhang/sos.html) reported in the paper, by run **benchmark_MSO.py**. It will automatically download the MSO dataset and the pre-trained VGG16 model.

The results are the same as the matlab implementation.

## Miscs
To change some configurations, please check **get_Param.py**.

There is an heuristic window refining process for small objects like in the matlab implementation. 
Note that this process is not included in the paper or used in the evaluation.

