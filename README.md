# TSBB19 Project 2: Visual Object Tracking

This toolkit provides a basic benchmark for visual object tracking
using the OTB-mini dataset.

## Setup
### Olympen computers
You can use the tsbb19 conda environment in the same way as for Project 1.

The dataset is located in the /courses/TSBB19/otb_mini folder, and should work with the provided skeleton code by default.

### Own computer
If you want to use your own computer, you need to copy the zipped otb-mini dataset from /courses/TSBB19/otb_mini.zip to wherever you need it.

You can e.g. use scp on the commandline, or the "download" feature in vscode.



## Code
### python\_tracker 
This folder contains python skeleton code for the tracking project.
It also contains some utility code that is useful.


## Evaluation

We will mainly use the **Success Rate** and **AUC** metrics. These metrics are related, and are both based on the per-image IoU scores for the boxes.
Code to produce these scores is given in python\_tracker/cvl/dataset.py and an example can be seen in python\_tracker/example_tracker.py.

### IoU
The IoU is defined per-image, as the area of intersection of the predicted and ground truth bounding box, divided by the area of the union. See python\_tracker/cvl/dataset.py for more details.
### Success Rate
For any given overlap threshold between [0,1], we can calculate the success rate as the number of frames that achieve a higher IoU than the threshold, divided by the total number of frames. See python\_tracker/cvl/dataset.py for more details.
### Area Under Curve (AUC)
AUC is a general performance measure which integrates an evaluation metric. In our case we are interested in the mean success rate over the possible overlap thresholds in the success rate metric. See python\_tracker/cvl/dataset.py for more details.


## Pitfalls

* Make sure to somehow deal with cases where the tracker goes outside the image, or else you will get errors.
* Think closely about the offsets calculated, note that negative offsets will be located in the far part of the correlation response if using circular correlation.
* For deep features, note that the resolution is lower than the original image, you will have to deal with this in some way.
* For deep features to run fast, you need to use cuda. We have provided an automatic version of this in python\_tracker/cvl/features_resnet.py, but you may want to change something, depending on your setup.