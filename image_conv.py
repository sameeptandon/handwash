import numpy as np
import sys, os
import cv2
from skimage.filter import *

def getFileNum(img_name):
  tokens = img_name.split("_")
  tokens = tokens[1].split(".")
  return int(tokens[0])

def getFileList(d, prefix):
  files = filter(lambda x: x.split('_')[0] == prefix, os.listdir(d))
  files.sort(key = getFileNum)
  files = map(lambda x: d + "/" + x, files)
  return files

data_dir = sys.argv[1]
inprefix = "img"
outprefix = "scharr"

inext = "jpg"
outext = "jpg"
img_files = getFileList(data_dir, inprefix)

for frame_count in range(len(img_files)):
  img_data = cv2.imread(img_files[frame_count])

  #kernel = np.ones((7,7),np.uint8)
  #img_data2 = cv2.erode(img_data,kernel,iterations = 1)
  img_data2 = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
  img_data2 = 255*scharr(img_data2)
  img_data2 = 255*img_data2 / np.max(img_data2)
  print img_data2

  out_fname = img_files[frame_count].replace(inprefix, outprefix)
  out_fname = out_fname.replace(inext, outext)
  print out_fname
  cv2.imwrite(out_fname, img_data2) 



