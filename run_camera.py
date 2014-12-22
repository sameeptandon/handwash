import cv2

c = cv2.VideoCapture(0)

while True: 
  success, img = c.read()
  if success:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    cv2.imshow('vid', img)


