from natsort import natsorted
from easydict import EasyDict as edict
import tensorflow as tf
from object_detection.utils import dataset_util
import argparse
import glob,os
from PIL import Image
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True,
                help="a las carpetas con las imagenes png y .reg")
ap.add_argument("--output_path", required=True,
                help="path al archivo .record")
FLAGS = ap.parse_args()

def create_tf_example(label_and_data_info):
  # TODO START: Populate the following variables from your example.
  height = label_and_data_info.height # Image height
  width = label_and_data_info.width # Image width
  filename = label_and_data_info.filename # Filename of the image. Empty if image is not from file
  encoded_image_data = label_and_data_info.encoded_image_data # Encoded image bytes
  image_format = label_and_data_info.image_format # b'jpeg' or b'png'

  xmins = label_and_data_info.xmins # List of normalized left x coordinates in bounding box (1 per box)
  xmaxs = label_and_data_info.xmaxs # List of normalized right x coordinates in bounding box
             # (1 per box)
  ymins = label_and_data_info.ymins # List of normalized top y coordinates in bounding box (1 per box)
  ymaxs = label_and_data_info.ymaxs # List of normalized bottom y coordinates in bounding box
             # (1 per box)
  classes_text = label_and_data_info.classes_text # List of string class name of bounding box (1 per box)
  classes = label_and_data_info.classes # List of integer class id of bounding box (1 per box)

  tf_label_and_data = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename),
      'image/source_id': dataset_util.bytes_feature(filename),
      'image/encoded': dataset_util.bytes_feature(encoded_image_data),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))
  return tf_label_and_data

def load_reg(path):
    fs = cv2.FileStorage(path,cv2.FILE_STORAGE_READ)
    rect = fs.getNode("rectangles")
    return rect.mat().reshape(-1,4)


def load_data_and_label(path):

    data_and_labels_files=[]
    lst_images = natsorted(glob.glob(path+"/*.png"))
    for fname in lst_images:
        config = edict()
        with tf.gfile.GFile(fname, 'rb') as fid:
            encoded_jpg = fid.read()

        image = Image.open(fname)
        config.height = image.height
        config.width = image.width
        config.filename = fname.encode('utf-8')
        config.encoded_image_data = encoded_jpg
        config.image_format = "png".encode('utf-8')

        reg_path = fname.split('.')[0]+".reg"
        if os.path.isfile(reg_path):
            BBs = load_reg(reg_path)
            config.xmins = [BB[0] for BB in BBs]
            config.ymins = [BB[1] for BB in BBs]
            config.xmaxs = [BB[0]+BB[2] for BB in BBs]
            config.ymaxs = [BB[1]+BB[3] for BB in BBs]
            config.classes_text = ["defecto".encode('utf-8') for BB in BBs]
            config.classes = [1 for BB in BBs]
        else:
            config.xmins = None
            config.ymins = None
            config.xmaxs = None
            config.ymaxs = None
            config.classes_text = None
            config.classes = None
        data_and_labels_files.append(config)
    return data_and_labels_files







def main(_):
  writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

  all_data_and_label_info = load_data_and_label(FLAGS.input)

  for data_and_label_info in all_data_and_label_info:
    tf_example = create_tf_example(data_and_label_info)
    writer.write(tf_example.SerializeToString())

  writer.close()

if __name__ == '__main__':
  tf.app.run()
