#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 10:17:55 2017

@author: abhay
"""

class CNN():
    DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
    # pylint: enable=line-too-long
    BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
    BOTTLENECK_TENSOR_SIZE = 2048
    MODEL_INPUT_WIDTH = 299
    MODEL_INPUT_HEIGHT = 299
    MODEL_INPUT_DEPTH = 3
    JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
    RESIZED_INPUT_TENSOR_NAME = 'ResizeBilinear:0'
    MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M
    
    root_dir = "/home/abhay/Documents/LearnersHeaven/videoindexing/"
    model_dir = root_dir+"inception"
    summaries_dir = ""
    intermediate_store_frequency = 0
    intermediate_output_graphs_dir=""

    def __init__(self):
        pass


class General():
    
    video_source="/home/deep-vision/Documents/datasets/videos/big_buck_bunny_720p_5mb.mp4"
    frames_taget_dir="/home/deep-vision/Documents/image-projects/videosearch/frames"

    
    def __init__(self):
        pass

import os
project_dir = "/home/abhay/Documents/LearnersHeaven/scene_search"
movie_dir = os.path.join(project_dir, "movie")
vector_dir = os.path.join(project_dir, "vectors")
frames_dir = os.path.join(project_dir, "frames")
metadata_dir = os.path.join(project_dir, "movie_metadata")
model_dir = os.path.join(project_dir, "model")