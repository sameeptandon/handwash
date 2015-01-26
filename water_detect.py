import numpy as np
import sys, os
import cv2
import skimage

def blue_filter(img_data):
  img_data2 = np.array(img_data)
  blue_low = 250
  blue_high = 255
  green_low = 0
  green_high = 210
  red_low = 0
  red_high = 210

  img_data2[ img_data[:,:,0] < blue_low, : ] = 0
  img_data2[ img_data[:,:,0] > blue_high, : ] = 0
  img_data2[ img_data[:,:,1] < green_low, : ] = 0
  img_data2[ img_data[:,:,1] > green_high, : ] = 0
  img_data2[ img_data[:,:,2] < red_low, : ] = 0
  img_data2[ img_data[:,:,2] > red_high, : ] = 0

  return img_data2
 
def blue_hist(img, bin_size=50):
  f = blue_filter(img)
  mask = f[:,:,0] > 0
  px, py = np.nonzero(mask)
  return np.histogram2d(px,py,bins=(720/bin_size, 1280/bin_size),range=[[0,720],[0,1280]])


def trainWaterDetector(data_dir, bin_size=10, cell_occupancy = 0.5):
  img_files = getFileList(data_dir, "img")
  TRAIN_IDX = range(5, len(img_files) / 2) 
  first = True
  for frame_count in TRAIN_IDX:
    img_data = cv2.imread(img_files[frame_count])
    if first:
      hist_data = blue_hist(img_data, bin_size=bin_size)[0] 
      first = False
    else:
      hist_data += blue_hist(img_data, bin_size=bin_size)[0]

  hist_data[ hist_data / (len(TRAIN_IDX) * bin_size**2) < cell_occupancy ] = 0
  hist_data[ hist_data > 0 ] = 1
  return hist_data


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

bin_size = 10
cell_occupancy = 0.5

bg_water_dir = sys.argv[1]
water_hist = trainWaterDetector(bg_water_dir, 
    bin_size = bin_size, 
    cell_occupancy = cell_occupancy)

water_hist_viz = cv2.resize(water_hist, (500, 500))
cv2.imshow("train", water_hist_viz/ np.max(water_hist_viz))
cv2.waitKey(50)

data_dir = sys.argv[2]
img_files = getFileList(data_dir, "img")
dep_files = getFileList(data_dir, "rawdepth")


for frame_count in range(5, len(img_files)):
  img_data = cv2.imread(img_files[frame_count])

  img_data2 = blue_filter(img_data)
  hist_data = blue_hist(img_data, bin_size=bin_size)[0] 
  hist_data[ hist_data / bin_size**2 < cell_occupancy ] = 0
  hist_data[ hist_data > 0 ] = 1

  water_locs = hist_data
  water_locs[water_hist == 0] = 0

  if (np.sum(water_locs) > np.sum(water_hist) / 10):
    print "water"
    cv2.putText(img_data, "water detected", (200, 200), cv2.FONT_HERSHEY_PLAIN, 5.0, (0,0,255))
  
 

  cv2.imshow("img", img_data)
  #cv2.imshow("img2", img_data2)
  cv2.imshow("water", cv2.resize(water_locs, (500,500)))
  #cv2.imshow("depth", depth_img / np.max(depth_img))
  cv2.waitKey(50)



