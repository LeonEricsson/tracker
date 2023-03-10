"""
Created by: Gustav Häger
Updated by: Johan Edstedt (2021)
"""


import argparse
import cv2

import numpy as np
from tqdm import tqdm
from cvl.dataset import OnlineTrackingBenchmark
from cvl.trackers import MOSSEtracker, NCCTracker, MOSSERGBtracker



if __name__ == "__main__":
    tuning = [1,2,3,4,5]
    test = [9, 15, 24, 17, 6, 12, 20, 19, 27]
    parser = argparse.ArgumentParser('Args for the tracker')
    parser.add_argument('--sequences',nargs="+",default=test,type=int)
    parser.add_argument('--dataset_path',type=str,default="/courses/TSBB19/otb_mini")
    parser.add_argument('--show_tracking',action='store_true',default=False)
    args = parser.parse_args()

    dataset_path,SHOW_TRACKING,sequences = args.dataset_path,args.show_tracking,args.sequences

    learning_rates = [0.8]
    #learning_rates = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for lr in learning_rates:
        dataset = OnlineTrackingBenchmark(dataset_path)
        results = []
        for sequence_idx in tqdm(sequences):
            a_seq = dataset[sequence_idx]

            if SHOW_TRACKING:
                cv2.namedWindow("tracker")
            tracker = MOSSEtracker(lr)
            pred_bbs = []
            for frame_idx, frame in tqdm(enumerate(a_seq), leave=False):
                image_color = frame['image']   
                image = np.sum(image_color, 2) / 3
                if type(tracker) == MOSSERGBtracker:
                    image = image_color
                if frame_idx == 0:
                    bbox = frame['bounding_box']
                    if bbox.width % 2 == 0:
                        bbox.width += 1

                    if bbox.height % 2 == 0:
                        bbox.height += 1
                    
                    current_position = bbox
                    tracker.start(image, bbox)
                    frame['bounding_box']
                else:
                    tracker.detect(image)
                    tracker.update(image)
                pred_bbs.append(tracker.get_bbox())
                if SHOW_TRACKING:
                    window = tracker.get_region()
                    bbox = tracker.get_bbox()
                    pt0 = (bbox.xpos, bbox.ypos)
                    pt1 = (bbox.xpos + bbox.width, bbox.ypos + bbox.height)
                    image_color = cv2.cvtColor(image_color, cv2.COLOR_RGB2BGR)
                    cv2.rectangle(image_color, pt0, pt1, color=(0, 255, 0), thickness=3)
                    pt0 = (window.xpos, window.ypos)
                    pt1 = (window.xpos + window.width, window.ypos + window.height)
                    cv2.rectangle(image_color, pt0, pt1, color=(255, 0, 0), thickness=1)
                    cv2.imshow("tracker", image_color)
                    cv2.waitKey(1)
            sequence_ious = dataset.calculate_per_frame_iou(sequence_idx, pred_bbs)
            results.append(sequence_ious)
        overlap_thresholds, success_rate = dataset.success_rate(results)
        auc = dataset.auc(success_rate)
        print(f'Tracker AUC: {auc}, Learning Rate: {lr}')
