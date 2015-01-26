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

background_dir = sys.argv[1]
background_depth_files = getFileList(background_dir, "rawdepth")
bg_depth_img = np.zeros((240,320))
#NUM_BACKGROUND_FILES = len(background_depth_files) / 2
NUM_BACKGROUND_FILES = 5
for frame_count in range(NUM_BACKGROUND_FILES):
  print frame_count
  depth_data = np.genfromtxt(background_depth_files[frame_count], delimiter=',', dtype=np.int32)
  depth_img = depthDataToImage(depth_data, shape=(240,320))
  cv2.imshow("bg", depth_img / 3000.0)
  cv2.waitKey(5)
  bg_depth_img = np.maximum(bg_depth_img, depth_img)

bg_depth_show = cv2.resize(bg_depth_img, (1280,960))
cv2.imshow("background", bg_depth_show / 3000)
cv2.waitKey(50)

data_dir = sys.argv[2]
img_files = getFileList(data_dir, "img")
dep_files = getFileList(data_dir, "rawdepth")
depth_img_ref = bg_depth_img

NUM_ZONE_FILES = 5
water_zone = np.zeros((240,320))
for frame_count in range(NUM_ZONE_FILES):
  img_data = cv2.imread(img_files[frame_count])
  depth_data = np.genfromtxt(dep_files[frame_count], delimiter=",", dtype=np.int32)

  depth_img = depthDataToImage(depth_data, shape=(240,320))

  print frame_count
  med = depth_img - depth_img_ref
  med[depth_img_ref == 0] = 0
  med[depth_img == 0] = 0
  med[med > -100] = 0
  med[depth_img > 500] = 0
  depth_img[med == 0] = 0
  water_zone = np.maximum(water_zone, depth_img)
  med = cv2.resize(med, (1280,960))
  
  cv2.imshow("data", -med / 800.0)
  cv2.imshow("data2", depth_img / 500.0)
  cv2.waitKey(5)
  
cv2.imshow("water_zone", water_zone / np.max(water_zone))
cv2.waitKey(50)
np.savetxt(sys.argv[3], water_zone)
