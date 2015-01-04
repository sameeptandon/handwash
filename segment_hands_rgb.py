import numpy as np
import sys, os
import cv2

def getFileNum(img_name):
  tokens = img_name.split("_")
  tokens = tokens[1].split(".")
  return int(tokens[0])

def getFileList(d, prefix):
  files = filter(lambda x: x.split('_')[0] == prefix, os.listdir(d))
  files.sort(key = getFileNum)
  files = map(lambda x: d + "/" + x, files)
  return files

def depthDataToImage(data, shape=(720,1280)):
  ret = np.zeros(shape)
  ret[data[:,1], data[:,0]] = data[:,2]
  return ret

def upsampleDepthImage(img): 
  window = 3;
  ret = np.array(img)
  for px in range(img.shape[0]):
    for py in range(img.shape[1]):
      if (img[px,py] != 0): continue

      subWindow = img[ max(0, px-window) : min(px + window, 719),
                       max(0, py-window) : min(py + window, 1279) ]
      nnz = subWindow[np.nonzero(subWindow)] 
      if len(nnz) > 0:
        ret[px,py] = np.min( nnz )
      else:
        ret[px,py] = 0
  return ret 

def upsampleDepthImageFast(img):
  window = 3;
  up = np.array(img);
  up[ up == 0 ] = 999999
  for wx in range(-window, window):
    for wy in range(-window, window):
      M = np.float32([[1,0,wx],[0,1,wy]])
      dst = cv2.warpAffine(img,M,(img.shape[1], img.shape[0]))
      dst[ dst == 0 ] = 999999
      up = np.minimum(up, dst)

  ret = np.array(img);
  ret[ret == 0] = up[ret == 0]
  ret[ret == 999999] = 0
  return ret

background_dir = sys.argv[2]
background_rgb_files = getFileList(background_dir, "img")
bg_rgb_img = np.zeros((720,1280,3))
NUM_BACKGROUND_FILES = 15
for frame_count in range(NUM_BACKGROUND_FILES):
  img = cv2.imread(background_rgb_files[frame_count])
  bg_rgb_img += img

bg_rgb_img /= NUM_BACKGROUND_FILES
cv2.imshow("background", bg_rgb_img / 255.0 )
cv2.waitKey(0)

data_dir = sys.argv[1]
img_files = getFileList(data_dir, "img")
dep_files = getFileList(data_dir, "img")
img_ref = bg_rgb_img

for frame_count in range(len(img_files)):
  img_data = cv2.imread(img_files[frame_count])
  print frame_count
  med = np.abs(img_data- img_ref)
  med = np.sum(med**2, axis=-1)**(1./2)
  print med
  med[med < 50] = 0
  print med.shape
  img_data[med == 0] = 0
  
  cv2.imshow("data", med / 255.0)
  cv2.imshow("img", img_data)
  cv2.waitKey(5)
  




