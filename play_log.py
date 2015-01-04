import numpy as np
import sys, os
import cv2
import skimage

def getFileNum(img_name):
  tokens = img_name.split("_")
  tokens = tokens[1].split(".")
  return int(tokens[0])

def getFileList(d, prefix):
  files = filter(lambda x: x.split('_')[0] == prefix, os.listdir(d))
  files.sort(key = getFileNum)
  files = map(lambda x: d + "/" + x, files)
  return files

def depthImageToData(img):
  nz_idx = np.nonzero(img)
  nnz = len(nz_idx[0])
  data = np.zeros((nnz, 3), dtype=np.int32)
  data[:,0] = nz_idx[1]
  data[:,1] = nz_idx[0]
  data[:,2] = img[nz_idx[0], nz_idx[1]]
  return data

def depthDataToImage(data, shape=(720,1280)):
  ret = np.zeros(shape)
  ret[data[:,1], data[:,0]] = data[:,2]
  return ret

def upsampleDepthImage(img):
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

data_dir = sys.argv[1]
img_files = getFileList(data_dir, "img")

for frame_count in range(len(img_files)):
  img_data = cv2.imread(img_files[frame_count])
  #depth_img = depthDataToImage(depth_data);
  #depth_img = upsampleDepthImage(depth_img);
  #depth_data = depthImageToData(depth_img)

  """
  gray = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
  sift = cv2.SURF()
  kp = sift.detect(gray)
  sift_img = cv2.drawKeypoints(gray,kp);
  cv2.imshow("sift", sift_img)
  """
  #kernel = np.ones((7,7),np.uint8)
  #img_data2 = cv2.erode(img_data,kernel,iterations = 1)
  #img_data2 = cv2.cvtColor(img_data2, cv2.COLOR_BGR2GRAY)
  img_data2 = cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
  from skimage.morphology import disk
  from skimage.filter.rank import gradient
  from skimage.filter import canny, sobel, roberts, scharr, gaussian_filter, threshold_adaptive, threshold_isodata, prewitt
  #img_data2 = gaussian_filter(img_data2, 1.0)
  img_data2 = canny(img_data2, 1.0)
  #img_data2[img_data2 > 0.05] = 1

  img_data_copy = np.array(img_data)
  #img_data[:,:,:] = 0
  #img_data[depth_data[:,1], depth_data[:,0], :] = img_data_copy[depth_data[:,1], depth_data[:,0], :]

  cv2.imshow("img", img_data)
  cv2.imshow("img2", img_data2 / (0.0 + np.max(img_data2)))
  cv2.waitKey(50)



