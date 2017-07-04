#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 01:10:14 2017

@author: abhay
"""

# import the necessary packages
from threading import Thread
import sys,os
import cv2

# import the Queue class from Python 3
if sys.version_info >= (3, 0):
    from queue import Queue

# otherwise, import the Queue class for Python 2.7
else:
    from Queue import Queue

class FileVideoStream:
    def __init__(self, path, queueSize=128):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.stream = cv2.VideoCapture(path)
        self.stopped = False
        self.image_name = None
        self.temporal_location = self.stream.get(cv2.CAP_PROP_POS_MSEC)
        self.sequence = self.stream.get(cv2.CAP_PROP_POS_FRAMES)
        # initialize the queue used to store frames read from
        # the video file
        self.Q = Queue(maxsize=queueSize)

    def start(self):
        # start a thread to read frames from the file video stream
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        # keep looping infinitely
        while True:
            # if the thread indicator variable is set, stop the
            # thread
            if self.stopped:
                return

            # otherwise, ensure the queue has room in it
            if not self.Q.full():
                # read the next frame from the file
                (grabbed, frame) = self.stream.read()

                current_index = self.stream.get(cv2.CAP_PROP_POS_FRAMES)

                self.sequence = current_index
                self.temporal_location = self.stream.get(cv2.CAP_PROP_POS_MSEC)
                self.image_name = "img-%d.jpg" % current_index

                # if the `grabbed` boolean is `False`, then we have
                # reached the end of the video file

                if not grabbed:
                    self.stop()
                    return
                # add the frame to the queue
                self.Q.put(frame)

    def get_frames_attributes(self):
        return {"temporal_location": self.temporal_location, "sequence": self.sequence}

    def read(self):
        # return next frame in the queue
        return self.Q.get()

    def more(self):
        # return True if there are still frames in the queue
        return self.Q.qsize() > 0

    def get_image_name(self):
        return self.image_name

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True


class FileStreamWrite:
    def __init__(self, image_path, data):
        self.image_path = image_path
        self.image_array = data
        self.stopped = False
        self.status = False

    def start(self):
        t = Thread(target=self.write_, args=())
        t.daemon = True
        t.start()
        return self

    def write_(self):
        while True:
            if self.stopped:
                return
            if not os.path.exists(self.image_path):
                cv2.imwrite(self.image_path, self.image_array)
                #self.status = True
        return self

    def written_frame_name(self):
        return self.image_path

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True
