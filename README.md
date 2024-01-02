# graduation
SEU graduation project

## Document description:

### Main code file:
  ui-t.py is the final running code
  
  ui-yolov5 design source code for running interface
  
  lane.py is the lane line detection code

### Minor code
(1) Kitti dataset preprocessing code
  
  d1_modify_ann.py
  
  d2_txt2xml.py
  
  d3_xml2txt.py
  
  d4_divdataset.py

(2) yolov5 training, testing, etc.
  
  detect.py detects code for yolov5
  
  train.py is the training code
  
  val.py is the model verification code

### Other
The data folder is used to save data, and the yaml files are used to process the dataset.

The model folder is used to save the model, and the yaml files are used to save the model's structure, and so on.

The run folder is used to save the results of training, validation, and testing.

The icon folder is used to save the icons that the program needs to use.

The other folders are the folders that come with the Yolov5 model.

## Lane Detection:
![Lane Detection Method](https://github.com/z-z-n/The-Vision-based-Driver-Assistance-System/blob/main/readme/LaneDetection.png)
