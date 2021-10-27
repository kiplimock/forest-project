## Forest Project
In this project, I will be making use of multiple view geometry to generate depth maps of scenes taken using a
stereoscopic camera. Using these depth maps, the objective is to identify areas with/out forest cover as well as compute various tree parameters such as height, crown size etc.

## Usage
#### Get images from single camera 
First create the a directory to save the images. Set the cameras in the code and then follow this example usage:

<code>python get_single_images.py image_dir 1</code>

#### Get images from stereo
Create a directory for the images and then two other directories for the right and left images. Example usage is:

`python get_stereo_images.py image_dir 1` 

#### Calibrate single camera
Your calibration images should be stored in a directory. Example usage:

`python single_calibration.py`

#### Calibrate stereo
Your camera coefficients for the left and right cameras should be saved in two files e.g. `right_cam.yml` and `left_cam.yml`. Example usage: 

`python stereo_camera_calibration.py --left_file left_cam.yml --right_file right_cam.yml --left_prefix left --right_prefix right --left_dir bothImagesFixedStereo --right_dir bothImagesFixedStereo --image_format png --square_size 0.025 --save_file stereo_cam.yml
`