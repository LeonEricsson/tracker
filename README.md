# Visual Object Trackers
Visual Object Trackers using the OTB-mini dataset. There are four available trackers: Normalized Cross-correlation, MOSSE Greyscale, MOSSE RGB and finally MOSSE Deep. All trackers were built following [tracking theory](https://www.sciencedirect.com/science/article/pii/B9780128221099000187) from Michael Felsberg. The MOSSE trackers have the same foundation but use different features for their discriminative filter where MOSSE Deep uses ResNet50 as a feature backbone. 


## Dataset
The dataset is a subset of the OTB dataset which can be found [here](https://paperswithcode.com/dataset/otb)

## Evaluation
We mainly use the **Success Rate** and **AUC** metrics. These metrics are related, and are both based on the per-image IoU scores for the boxes.
Code to produce these scores is given in python\_tracker/cvl/dataset.py and an example can be seen in python\_tracker/example_tracker.py.

### IoU
The IoU is defined per-image, as the area of intersection of the predicted and ground truth bounding box, divided by the area of the union. See python\_tracker/cvl/dataset.py for more details.
### Success Rate
For any given overlap threshold between [0,1], we can calculate the success rate as the number of frames that achieve a higher IoU than the threshold, divided by the total number of frames. See python\_tracker/cvl/dataset.py for more details.
### Area Under Curve (AUC)
AUC is a general performance measure which integrates an evaluation metric. In our case we are interested in the mean success rate over the possible overlap thresholds in the success rate metric. See python\_tracker/cvl/dataset.py for more details.

