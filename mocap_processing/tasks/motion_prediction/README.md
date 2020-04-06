# Motion Prediction

Human motion prediction is the problem of forecasting future body poses given observed pose sequence. 

## Task
### Input
Sequences of 2 seconds length (120 frames)
### Output
Short term Prediction: Sequences of 400-ms (24 frames)
Long term Prediction: Sequences of 1000-ms (60 frames)

### Test set
H3.6M benchmarks use a total of 120 test samples across 15 categories
AMASS benchmark uses 3304 test samples (5% of dataset)

### Metrics
Mean (Euler) Angle Error of poses at 80, 160, 320 and 400 ms

## Commands
### Preprocessing
AMASS data can be downloaded from this link http://dip.is.tue.mpg.de/ where sequences are stored as `.pkl` files. We use `amass_dip` loader to load raw data in Motion objects, extract sequence windows, represent them as list of (source, target) tuples in their matrix versions, and split into training, validation and test sets.
```
python mocap_processing/tasks/motion_prediction/preprocess.py --input-dir <PATH TO RAW DATA> --output-dir <PREPROCESSED OUTPUT PATH> --split-dir ./mocap_processing/tasks/motion_prediction/data/ --rep aa
```
### Training
```
python mocap_processing/tasks/motion_prediction/training.py --save-model-path <PATH TO SAVE MODELS>  --preprocessed-path <PREPROCESSED DATA PATH> --epochs 70
```
### Test
```
python mocap_processing/tasks/motion_prediction/test.py --save-model-path <PATH TO MODEL> --preprocessed-path <PREPROCESSED DATA PATH> --save-output-path <PATH TO SAVE PREDICTED MOTIONS>
```
