# Test-Yolo-On-Image

The weights file is trained for using the model by cameras placed at a lower level angle (parking entrance), but, works as well in different situations.
The cfg file contains a custom config for a single class object detection model.
The names file contain a tag that model is trained for, only one object for this weights file(plateNumber).

python script uses OpenCV and Numpy to detect object(s) depends on your weights file, in this situation, plate numbers. Can use for other trained models.
