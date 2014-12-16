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
background_depth_files = getFileList(background_dir, "rawdepth")
bg_depth_img = np.zeros((240,320))
NUM_BACKGROUND_FILES = len(background_depth_files) / 2
for frame_count in range(NUM_BACKGROUND_FILES):
  depth_data = np.genfromtxt(background_depth_files[frame_count], delimiter=',', dtype=np.int32)
  depth_img = depthDataToImage(depth_data, shape=(240,320))
  cv2.imshow("bg", depth_img / 3000.0)
  cv2.waitKey(5)
  bg_depth_img = np.maximum(bg_depth_img, depth_img)

bg_depth_show = cv2.resize(bg_depth_img, (1280,960))
cv2.imshow("background", bg_depth_show / 3000)
cv2.waitKey(50)

data_dir = sys.argv[1]
img_files = getFileList(data_dir, "img")
dep_files = getFileList(data_dir, "rawdepth")
depth_img_ref = bg_depth_img

for frame_count in range(len(img_files)):
  img_data = cv2.imread(img_files[frame_count])
  depth_data = np.genfromtxt(dep_files[frame_count], delimiter=",", dtype=np.int32)

  #img_data_copy = np.array(img_data)
  #img_data[:,:,:] = 0
  #img_data[depth_data[:,1], depth_data[:,0], :] = img_data_copy[depth_data[:,1], depth_data[:,0], :]

  depth_img = depthDataToImage(depth_data, shape=(240,320))
  #depth_img = upsampleDepthImageFast(depth_img)

  print frame_count
  depth_img[depth_img > 500] = 0
  med = np.abs(depth_img - depth_img_ref)
  med[depth_img == 0] = 0
  med[med < 100] = 0
  #med[med > 700] = 0
  depth_img[med == 0] = 0
  med = cv2.resize(med, (1280,960))
  
  cv2.imshow("data", med / 800.0)
  cv2.imshow("data2", depth_img / 500.0)
  cv2.waitKey(5)
  
  """
  depth_img[depth_img > 4000] = 0
  depth_img = cv2.resize(depth_img, (1280,960))
  #cv2.imshow("img", img_data)
  cv2.imshow("data", depth_img / 4000.0)
  cv2.waitKey(5)
  """



