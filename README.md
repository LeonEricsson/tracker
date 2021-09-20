# TSBB19 Project 2: Visual Object Tracking

This toolkit provides a basic benchmark for visual object tracking
using the OTB-mini dataset.

##  python\_tracker 
This folder contains python skeleton code for the tracking project.
It also contains some utility code that is useful.

## Evaluation

We will mainly use the **success rate** and **AUC** metrics. These metrics are related, and are both based on the per-image IoU scores for the boxes.
Code to produce these scores is given in python\_tracker/cvl/dataset.py and an example can be seen in python\_tracker/example_tracker.py.

### IoU
The IoU is defined per-image, as the area of intersection of the predicted and ground truth bounding box, divided by the area of the union. See python\_tracker/cvl/dataset.py for more details.
### Success Rate
For any given overlap threshold between [0,1], we can calculate the success rate as the number of frames that achieve a higher IoU than the threshold, divided by the total number of frames. See python\_tracker/cvl/dataset.py for more details.
### Area Under Curve (AUC)
AUC is a general performance measure which integrates an evaluation metric. In our case we are interested in the mean success rate over the possible overlap thresholds in the success rate metric. See python\_tracker/cvl/dataset.py for more details.
