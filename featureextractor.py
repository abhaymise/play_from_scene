import cv2
import time
import imutils
from imutils.video import FPS
import os,sys
import tensorflow as tf
import encoding as cnn
import numpy as np
from collections import defaultdict
from ThreadeIO import FileStreamWrite,FileVideoStream
import json
import configuration as cfg

def play(path,offset):
    stream = cv2.VideoCapture(path)
    stream.set(cv2.CAP_PROP_POS_MSEC,offset-5000)
    while True:
        ok,frame = stream.read()

        if not ok:
            break
        frame = imutils.resize(frame, width=450)
        #frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

        cv2.imshow("frame",frame)
        cv2.waitKey(30)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    stream.release()
    cv2.destroyAllWindows()

def file_already_existing(frame_url):
    return os.path.exists(frame_url)

def read_image_as_jpeg_encoded_string(image_path):
    if not tf.gfile.Exists(image_path):
        tf.logging.fatal('File does not exist %s', image_path)
    image_data = tf.gfile.FastGFile(image_path, 'rb').read()
    return image_data

def get_image_encoding(frame_path, sess, jpeg_data_tensor, bottleneck_tensor):
    image_data = read_image_as_jpeg_encoded_string(frame_path)
    bottleneck_tensor_out = cnn.run_bottleneck_on_image(sess, image_data,
                                                        jpeg_data_tensor, bottleneck_tensor)
    return bottleneck_tensor_out


def write_image_encoding(array, array_name, array_dir):
    parent_dir = cfg.vector_dir
    movie_dir = os.path.join(parent_dir, array_dir)
    cnn.ensure_dir_exists(movie_dir)
    encoded_vector_path = os.path.join(movie_dir, array_name)
    if not file_already_existing(encoded_vector_path):
        np.save(encoded_vector_path, array)
        print('\r>> Writing %s to location %s%%' %
              (array_name, encoded_vector_path))
    else:
        print " Image encoding for %s has been already witten " % encoded_vector_path


def _write_movie_metadata(json_content, json_name):
    parent_dir = cfg.metadata_dir
    cnn.ensure_dir_exists(parent_dir)
    movie_json_path = os.path.join(parent_dir, json_name) + ".json"
    if not file_already_existing(movie_json_path):
        with open(movie_json_path, 'w') as fp:
            json.dump(json_content, fp)
    else:
        print " JSON for %s already exists " % movie_json_path

def get_movie_name_ext(movie_basename):
    movie_detail = movie_basename.rsplit('.')
    movie_name,movie_extension = movie_detail[0],movie_detail[1]
    assert len(movie_extension)==3
    return movie_name,movie_extension

def get_movie_directory_from_movie_url(movie_url):
    video_extension = ['avi','mp4']
    movie_basename = os.path.basename(movie_url)
    name,movie_extension = get_movie_name_ext(movie_basename)
    if movie_extension not in video_extension :
        print " movie format not supported "
        return
    return name


def read_write_frames(video):
    fvs = FileVideoStream(video).start()
    time.sleep(1.0)

    # start the FPS timer
    fps_read = FPS().start()
    fps_write = FPS().start()
    movie_name = get_movie_directory_from_movie_url(video)
    image_dir = os.path.join(cfg.frames_dir,movie_name)

    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    count = 0
    image_meta = defaultdict(list)
    written_frame_names=set()
    # loop over frames from the video file stream
    while fvs.more():
        # grab the frame from the threaded video file stream, resize
        # it, and convert it to grayscale (while still retaining 3
        # channels)
        frame = fvs.read()
        print fvs.get_frames_attributes()
        count += 1
        #frame = imutils.resize(frame, width=450)
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #frame = np.dstack([frame, frame, frame])
        image_name = fvs.get_image_name()
        image_full_url = os.path.join(image_dir, image_name)

        meta = fvs.get_frames_attributes()
        meta["name"] = image_name
        image_meta[movie_name].append(meta)

        # display the size of the queue on the frame
        #cv2.putText(frame, "Queue Size: {}".format(fvs.Q.qsize()),
        #            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        # show the frame and update the FPS counter
        cv2.imshow("Frame", frame)

        fps_read.update()

        abc = FileStreamWrite(image_full_url, frame).start()
        written_frame_names.add(abc.written_frame_name())
        fps_write.update()

#        abc.stop()
        cv2.waitKey(1)
    # stop the timer and display FPS information
    fps_read.stop()
    fps_write.stop()
    _write_movie_metadata(image_meta, movie_name)
    print("[INFO] elasped time: {:.2f}".format(fps_read.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps_write.fps()))

    assert fps_read._numFrames == fps_write._numFrames
    return written_frame_names

def cache_bottlenecks(video):
    """
    This reads the frames from a video and constructs and
    returns a list of (name-%d.jpg,time) pair
    """
    cnn.maybe_download_and_extract()

    movie_name = get_movie_directory_from_movie_url(video)
    graph, bottleneck_tensor, jpeg_data_tensor, _ = (
        cnn.create_inception_graph())
    frame_list = read_write_frames(video)
    with tf.Session(graph=graph) as sess:
            for frame_path in frame_list:
                bottleneck_tensor_out = get_image_encoding(frame_path, sess, jpeg_data_tensor, bottleneck_tensor)
                image_encoding_name = os.path.basename(frame_path) + ".npy"
                write_image_encoding(bottleneck_tensor_out, image_encoding_name, movie_name)


if __name__=="__main__":
    cache_bottlenecks("/home/abhay/Documents/LearnersHeaven/scene_search/movie/ShapeOfYou.mp4")