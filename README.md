# crashtest_targetmarker_detection_yquad

## 1. Create conda environment and install requirements
`conda create env -n yquad python=3.9.* pip`

`conda activate yquad`

`pip install -r requirements.txt`

## 2. Edit video.cfg
This is the config file where input/output paths etc. have to be defined.

The output directory will be created automatically if it does not already exist. More details to each parameter can be found in the config file itself.

## 3. Run program
`python process_video.py`
