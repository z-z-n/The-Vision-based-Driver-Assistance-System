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
As shown in Figure 3-1 detection algorithm diagram, the input image will output the lane line information and detection result image after 3 steps. image.

The first step is preprocessing, after the input image is filtered by color filtering, L-channel filtering and Sobel's algorithm with a total of 3 thresholds, a grayscale binary image will be obtained. After the input image is filtered by color filtering, L-channel filtering and Sobel algorithm with three thresholds, a gray-scale binary image is obtained; the gray-scale binary image is transformed by IPM inverse perspective, and the view angle is transformed from front view to "bird's eye view"; the gray-scale binary image is transformed by IPM inverse perspective, and the view angle is transformed by IPM inverse perspective. The gray-scale binary image will be obtained after IPM inverse perspective transformation, and the viewpoint will be changed from front view to "bird's-eye view"; the bird's-eye view will be processed by the region of interest to obtain the cleaner lane line binary image.

The sliding window stage utilizes the pre-processed binary map to obtain the lane line information. Based on the previous binary map, first obtain The initial histogram is obtained to determine the initial position of the sliding window; the sliding window searches for the left and right lane lines according to the initial position. The sliding window is then slid according to the initial position to search for the left and right lane lines respectively; after that, the valid pixel points are obtained to fit the left and right lane lines.

In the post-processing part, the radius of curvature of the lane lines is calculated by using the coefficients of the fitted curves in combination with the multi-frame averaging method, The post-processing part utilizes the fitted curve coefficients and the multi-frame averaging method to calculate the radius of curvature of the lane lines, the sparsity, etc., and maps the fitted curves and lane line areas to the original image.

https://github.com/z-z-n/The-Vision-based-Driver-Assistance-System/assets/65029895/448ad4dd-a159-4fdf-a086-face54a5ec43

