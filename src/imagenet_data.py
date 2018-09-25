from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import cv2
import numpy as np
from Queue import Queue
from threading import Thread as Process

from skimage.transform import color

# from utils import *

class DataSet(object):
  """TextDataSet
  process text input file dataset 
  text file format:
    image_path
  """

  def __init__(self, data_path='/home/xieya/colorization-tf/data/train.txt', nthread=12):
    """
    Args:
      common_params: A dict
      dataset_params: A dict
    """
    self.image_size = 256
    self.batch_size = 16
    self.data_path = data_path
    self.thread_num = nthread
    self.thread_num2 = nthread
    self.record_queue = Queue(maxsize=30000)
    self.image_queue = Queue(maxsize=15000)
    self.batch_queue = Queue(maxsize=300)
    self.record_list = []  

    # filling the record_list
    input_file = open(self.data_path, 'r')

    for line in input_file:
      line = line.strip()
      self.record_list.append(line)

    self.record_point = 0
    self.record_number = len(self.record_list)

    self.num_batch_per_epoch = int(self.record_number / self.batch_size)

    t_record_producer = Process(target=self.record_producer)
    t_record_producer.daemon = True
    t_record_producer.start()

    for i in range(self.thread_num):
      t = Process(target=self.record_customer)
      t.daemon = True
      t.start()

    for i in range(self.thread_num2):
      t = Process(target=self.image_customer)
      t.daemon = True
      t.start()

  def record_producer(self):
    """record_queue's processor
    """
    while True:
      if self.record_point % self.record_number == 0:
        random.shuffle(self.record_list)
        self.record_point = 0
      self.record_queue.put(self.record_list[self.record_point])
      self.record_point += 1

  def image_process(self, image):
    """record process 
    Args: record 
    Returns:
      image: 3-D ndarray
    """
    h = image.shape[0]
    w = image.shape[1]

    if w > h:
      image = cv2.resize(image, (int(self.image_size * w / h), self.image_size))

      mirror = np.random.randint(0, 2)
      if mirror:
        image = np.fliplr(image)
      crop_start = np.random.randint(0, int(self.image_size * w / h) - self.image_size + 1)
      image = image[:, crop_start:crop_start + self.image_size, :]
    else:
      image = cv2.resize(image, (self.image_size, int(self.image_size * h / w)))
      mirror = np.random.randint(0, 2)
      if mirror:
        image = np.fliplr(image)
      crop_start = np.random.randint(0, int(self.image_size * h / w) - self.image_size + 1)
      image = image[crop_start:crop_start + self.image_size, :, :]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

  def record_customer(self):
    """record queue's customer 
    """
    while True:
      item = self.record_queue.get()
      out = cv2.imread(item)
      if len(out.shape)==3 and out.shape[2]==3:
        self.image_queue.put(out)

  def image_customer(self):
    while True:
      images = []
      for i in range(self.batch_size):
        image = self.image_queue.get()
        image = self.image_process(image)
        images.append(image)
      images = np.asarray(images, dtype=np.uint8)

      # RGB to LAB
      imgs_lab = color.rgb2lab(images)
      # imgs_lab[:, :, :, 0: 1] /= 50.
      # imgs_lab[:, :, :, 0: 1] -= 1.
      # imgs_lab[:, :, :, 1:] /= 110.

      self.batch_queue.put(imgs_lab)

  def batch(self):
    """get batch
    Returns:
      images: 4-D ndarray [batch_size, height, width, 3]
    """
    # print(self.record_queue.qsize(), self.image_queue.qsize(), self.batch_queue.qsize())
    return self.batch_queue.get()
