#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 13:39:27 2017

@author: abhay
"""
from time import sleep

import featureextractor as feat
import configuration as cfg
import numpy as np
import cv2
import encoding as cnn
import tensorflow as tf
import json
import os


def encode_query_image(test_image):
    graph, bottleneck_tensor, jpeg_data_tensor, _ = (
          cnn.create_inception_graph())
    with tf.Session(graph=graph) as sess:
        test_embedding = feat.get_image_encoding(test_image,sess,jpeg_data_tensor,bottleneck_tensor)
    return test_embedding.reshape(1,-1)


def get_list_of_embdedding(embedding_dir):
    list_name = [npy for npy in os.listdir(embedding_dir) if npy.endswith(".npy")]
    list_vector = [np.load(os.path.join(embedding_dir,name)) for name in list_name]
    assert len(list_name) == len(list_vector)
    return  np.array(list_vector,dtype=np.float32),np.array(list_name)
    
def get_neighbor(image_vec,existing_vectors,existing_vector_names):
    from neighbor import KNearestNeighbor
    classifier = KNearestNeighbor(existing_vectors,existing_vector_names)
    dists = classifier.compute_distances_no_loops(image_vec)
    y_test_pred = classifier.predict_labels(dists, k=1)
    return y_test_pred[0]

def load_movie_meta_data(movie_url):
    #from pprint import pprint
    movie_name = feat.get_movie_directory_from_movie_url(movie_url)
    movie_meta_data_path = os.path.join(cfg.metadata_dir , movie_name.rsplit(".")[0]+".json")

    with open(movie_meta_data_path,"r") as data_file:
        data = json.load(data_file)
    return data[movie_name]

def get_neighbors_temporal_location(movie_meta,neighbor):
    location=None
    for obj in movie_meta:
        if obj['name'] in neighbor:
            location = obj['temporal_location']
            #print location
            #print obj['name']
    return location

def find_and_play(test_image,movie_url):
    movie_name = feat.get_movie_directory_from_movie_url(movie_url)
    test_image_vector = encode_query_image(test_image)
    vector_dir_url = os.path.join(cfg.vector_dir, movie_name)
    movie_vectors, vector_names = get_list_of_embdedding(vector_dir_url)
    # print vector_names
    matched_image = get_neighbor(test_image_vector, movie_vectors, vector_names)
    print "matched neighbor ",matched_image
    movie_metadata = load_movie_meta_data(movie_url)
    location = get_neighbors_temporal_location(movie_metadata, matched_image)
    print location
    if location is not None:
        feat.play(movie_url, location)

if __name__=="__main__":
    #test_image = "/home/abhay/Documents/LearnersHeaven/scene_search/frames/jurassic_park_intro/img-634.jpg"
    #movie_url = "/home/abhay/Documents/LearnersHeaven/scene_search/movie/jurassic_park_intro.mp4"

    import tkFileDialog

    movie_url = tkFileDialog.askopenfilename(title="SELECT THE MOVIE !!! ")

    test_image = tkFileDialog.askopenfilename(title="SELECT THE SCENE !!! ")

    #print "query image ",test_image

    import imutils
    image = cv2.imread(test_image)
    image = imutils.resize(image, width=450)
    cv2.imshow("Query Image",image)
    cv2.waitKey(3000)

    find_and_play(test_image, movie_url)



    
    
    
    
    