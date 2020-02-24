# Adpative Patch Selection

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/) *

As part of our implementation of the A-Lamp paper, this is my version of the Adaptive Patch Selection algorithm, pre-prossing the images for the Multi-Patch Selection Subnet.

> [Shuang Ma, Jing Liu and Chang Wen Chen. "A-Lamp: Adaptive Layout-Aware Multi-Patch Deep Convolutional Neural Network for Photo Aesthetic Assessment." 	arXiv:1704.00248, 2017.](https://arxiv.org/abs/1704.00248)

The goal is to select areas of interest in an image, following a series of critera described in the paper. Please cite the above paper if you are to use this code.

![Alt text](girl_patches.png?raw=true "Example of patches, to be compared with results from the paper.")

## Prerequisites
1. Linux
2. Python 3 
3. Open CV 4.1
4. Scipy 1.4

For detailed prerequisites, please refer to AdaptivePatchSelection_CodeDocumentation.pdf.

## Quick Start

Run **./patch_selection.py \<images folder\> \<output folder\>**.

## Results on the AVA Dataset

Patches of images from the AVA Dataset are stored in the output-split in pickle files. For details, please refer to AdaptivePatchSelection_CodeDocumentation.pdf.

## 

\* This implementation is under MIT License, except for the algorithm used for computing saliency which is not to be used for commercial purposes. For more information, please refere to AdaptivePatchSelection_CodeDocumentation.pdf.
