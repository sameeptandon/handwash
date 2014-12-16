#!/usr/bin/python
# -*- coding: utf-8 -*-

from VtkRenderer import *
import numpy as np
import sys
import os
import time

global count
count = 0

def RGBDToXYZRGB(img, depth_data):
  KK = np.array([[970, 0, 640],
                 [0, 977, 480],
                 [0, 0, 1]])

  depth = np.zeros((depth_data.shape[0], 3))
  depth[:, 0:3] = depth_data[:, 0:3]
  depth[:, 2] /= 1000.0 #convert to meters

  pts = np.dot(np.linalg.inv(KK), depth.T)

  XYZRGB = np.zeros((depth_data.shape[0], 6));
  XYZRGB[:, 0:3] = pts.T
  XYZRGB[:, 3:6] = img[depth_data[:, 1], depth_data[:,0], :]

  return XYZRGB

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

def depthImageToData(img):
  nz_idx = np.nonzero(img)
  nnz = len(nz_idx[0])
  data = np.zeros((nnz, 3), dtype=np.int32)
  data[:,0] = nz_idx[1]
  data[:,1] = nz_idx[0]
  data[:,2] = img[nz_idx[0], nz_idx[1]]
  return data

def upsampleDepthImage(img):
  window = 5;
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


class GrabberCallback:

    def __init__(self):
        pass

    def execute(self, obj, event):
        global count
        t = time.time()
        if count >= len(img_files):
            return

        img_data = cv2.imread(img_files[count])
        depth_data = np.genfromtxt(dep_files[count], delimiter=",", dtype=np.int32)
        depth_img = depthDataToImage(depth_data)
        depth_img = upsampleDepthImage(depth_img)
        depth_data = depthImageToData(depth_img)

        XYZRGB = RGBDToXYZRGB(img_data, depth_data)
        count += 1
        iren = obj
        self.pointCloud = VtkPointCloud(XYZRGB[:, 0:3], XYZRGB[:, 3:6])
        actor = self.pointCloud.get_vtk_color_cloud()
        if count > 1:
            iren.GetRenderWindow().GetRenderers().GetFirstRenderer().RemoveActor(self.actor)
        self.actor = actor
        iren.GetRenderWindow().GetRenderers().GetFirstRenderer().AddActor(self.actor)
        if count == 1:
            iren.GetRenderWindow().GetRenderers().GetFirstRenderer().ResetCamera()

        iren.GetRenderWindow().Render()
        print time.time() - t


if __name__ == '__main__':
    data_dir = sys.argv[1]
    img_files = getFileList(data_dir, "img")
    dep_files = getFileList(data_dir, "depth")

    renderer = vtk.vtkRenderer()
    renderer.SetBackground(0., 0., 0.)
    renderer.ResetCamera()

    # Render Window

    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)
    renderWindow.SetSize(1280, 720)

    # Interactor

    renderWindowInteractor = vtk.vtkRenderWindowInteractor()
    renderWindowInteractor.SetRenderWindow(renderWindow)
    mouseInteractor = vtk.vtkInteractorStyleTrackballCamera()
    renderWindowInteractor.SetInteractorStyle(mouseInteractor)
    renderWindow.Render()

    cb = GrabberCallback()
    renderWindowInteractor.AddObserver('TimerEvent', cb.execute)
    timerId = renderWindowInteractor.CreateRepeatingTimer(500)
    renderWindowInteractor.Start()
