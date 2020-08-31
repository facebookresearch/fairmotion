# Motion Prediction

Human motion prediction is the problem of forecasting future body poses given observed pose sequence. The problem requires encoding both spatial and temporal aspects of human motion, and generating sequences conditioned on that. The problem is formulated as a sequence modeling task -- the input is 120 poses (frames) i.e., 2s of motion at 60Hz, and the output is 24 poses (400ms).

## Data
We use the AMASS DIP dataset which contains 42 hours of recorded human motion data. AMASS benchmark uses 3304 test samples (5% of dataset). See Preprocessing section for instructions on downloading and loading the dataset.

## Metrics
Mean (Euler) Angle Error of poses at 80, 160, 320 and 400 ms. Mean Angle Error is the Euclidean distance between predicted and reference Euler angles averaged over all joints, summed up to N frames. See `metrics.euler_diff` for code that computes this metric.

## Preprocessing
AMASS data can be downloaded from this link http://dip.is.tue.mpg.de/ where sequences are stored as `.pkl` files. We use `amass_dip` loader to load raw data in Motion objects, extract sequence windows, represent them as list of (source, target) tuples in their matrix versions, and split into training, validation and test sets. The preprocessing steps and dataset splits have been proposed by *Structured Prediction Helps 3D Human Motion Modelling*, Aksan et al ([source code](https://github.com/eth-ait/spl)). The following angle representations can be used -- axis angles (`aa`), quaternion (`quat`) and rotation matrix (`rotmat`).
```
python fairmotion/tasks/motion_prediction/preprocess.py \
    --input-dir <PATH TO RAW DATA> \
    --output-dir <PREPROCESSED OUTPUT PATH> \
    --split-dir ./fairmotion/tasks/motion_prediction/data/ \
    --rep aa
```
## Training
To build models for the motion prediction task, we provide several architectures -- RNN, seq2seq, tied seq2seq, transformer and transformer-LSTM hybrid models. The training script trains models for specified number of epochs while saving models every `--save-model-frequency` epochs and also the best performing on validation set.
```
python fairmotion/tasks/motion_prediction/training.py \
    --save-model-path <PATH TO SAVE MODELS> \
    --preprocessed-path <PREPROCESSED DATA PATH> \
    --epochs 100
```
## Test
The test script loads the best saved model from `--save-model-path`, and evaluates its performance on the test set. `--save-output-path` can be used to generate and save output sequences for a subset of the test set.
```
python fairmotion/tasks/motion_prediction/test.py \
    --save-model-path <PATH TO MODEL> \
    --preprocessed-path <PREPROCESSED DATA PATH> \
    --save-output-path <PATH TO SAVE PREDICTED MOTIONS>
```
