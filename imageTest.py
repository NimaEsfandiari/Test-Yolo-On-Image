import cv2
import numpy as np

# Load Yolo
net = cv2.dnn.readNet("yolov4-tiny-plateNumber.weights", "yolov4-tiny-obj.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layerNames = net.getLayerNames()
output_layers = [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Loading image
image = cv2.imread("testImages/x6.jpg")
height, width, channels = image.shape

# Detecting object
blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)


net.setInput(blob)
outs = net.forward(output_layers)

# Showing information on the screen
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:

            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Rectangle coordinate
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)


cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
