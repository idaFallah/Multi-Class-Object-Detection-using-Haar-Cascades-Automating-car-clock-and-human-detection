import cv2

from google.colab import drive
drive.mount('/content/drive')

from google.colab.patches import cv2_imshow

car_detector = cv2.CascadeClassifier('/content/drive/MyDrive/Computer Vision/Cascades/cars.xml')
image = cv2.imread('/content/drive/MyDrive/Computer Vision/Images/car.jpg')

image.shape

cv2_imshow(image) # image is small no need for resizing

#image = cv2.resize(image,(600, 400)) resizing the image to be bigger than its original size was a bad idea

image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
car_detections = car_detector.detectMultiScale(image_gray, scaleFactor=1.005, minNeighbors=6, maxSize=(60, 60), minSize=(40, 40))

for (x, y, w, h) in car_detections:
  print(w, h)
  cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2_imshow(image)

# clock detection

clock_detector = cv2.CascadeClassifier('/content/drive/MyDrive/Computer Vision/Cascades/clocks.xml')
image = cv2.imread('/content/drive/MyDrive/Computer Vision/Images/clock.jpg')
#cv2_imshow(image) # image is small no need for resizing
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
clock_detections = clock_detector.detectMultiScale(image_gray, scaleFactor=1.021)

for (x, y, w, h) in clock_detections:
  print(w, h)
  cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2_imshow(image)

# detecting full body of humans

fullBody_detector = cv2.CascadeClassifier('/content/drive/MyDrive/Computer Vision/Cascades/fullbody.xml')
image = cv2.imread('/content/drive/MyDrive/Computer Vision/Images/people3.jpg')
#cv2_imshow(image) # image is small no need for resizing
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
fullBody_detections = fullBody_detector.detectMultiScale(image_gray, scaleFactor=1.1)

for (x, y, w, h) in fullBody_detections:
  print(w, h)
  cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2_imshow(image)