# Tensorflow-Eyewear-and-HAR-with-CNN
HAR and eyewear's accelerometer :  from data acquisition to frozen model.
Valentin Morel - Ellcie Healthy SAS 

This project was about the HAR with eyewear accelerometer. You'll find in this repository all the files from end to end. 

all the raw data : 

9 subjects x 21 min (per subject) on 6 activities (Sitting, Standing, Standing up, Walking, Walking upstairs, Walking downstairs)


algorithm training : cnn_wism_no_features.py

gives the :
model_cnn-15000;
checkpoint;
model_cnn-15000.meta;

The test python file is : restore_cnn_model.py


The saved files give then the : frozen_model.pb



Finally a test with the frozen model, the normalization... was performed : restore_frozen.py


To execute a file, you just need to have everything in the same path. Nothing more ! Basic use.


Note : In the CNN, gyroscope data was first integrated. But it was dramatically bad. 
