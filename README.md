# crashtest_targetmarker_detection_yquad
![visualization](https://github.com/DataScienceLabFHSWF/crashtest_targetmarker_detection_yquad/blob/main/08_03_152.jpg)
![visualization](https://github.com/DataScienceLabFHSWF/crashtest_targetmarker_detection_yquad/blob/main/hyu_01_8.jpg)

## 1. Create conda environment and install requirements
`conda create env -n yquad python=3.9.* pip`

`conda activate yquad`

`pip install -r requirements.txt`

## 2. Edit video.cfg
This is the config file where input/output paths etc. have to be defined.

The output directory will be created automatically if it does not already exist. More details about each parameter can be found in the config file itself.

## 3. Run program
`python process_video.py`
