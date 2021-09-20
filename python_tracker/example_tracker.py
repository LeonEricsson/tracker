#!/usr/bin/env python3

import cv2

import numpy as np

from cvl.dataset import OnlineTrackingBenchmark
from cvl.trackers import NCCTracker

dataset_path = "/courses/TSBB19/otb_mini"

SHOW_TRACKING = False
SEQUENCE_IDX = 1

if __name__ == "__main__":

    dataset = OnlineTrackingBenchmark(dataset_path)

    a_seq = dataset[SEQUENCE_IDX]

    if SHOW_TRACKING:
        cv2.namedWindow("tracker")

    tracker = NCCTracker()
    pred_bbs = []
    for frame_idx, frame in enumerate(a_seq):
        print(f"{frame_idx} / {len(a_seq)}")
        image_color = frame['image']
        image = np.sum(image_color, 2) / 3
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
        pred_bbs.append(tracker.get_region())
        if SHOW_TRACKING:
            bbox = tracker.get_region()
            pt0 = (bbox.xpos, bbox.ypos)
            pt1 = (bbox.xpos + bbox.width, bbox.ypos + bbox.height)
            image_color = cv2.cvtColor(image_color, cv2.COLOR_RGB2BGR)
            cv2.rectangle(image_color, pt0, pt1, color=(0, 255, 0), thickness=3)
            cv2.imshow(f"tracker{frame_idx}.jpg", image_color)
            cv2.waitKey(0)
    print(dataset.calculate_per_frame_iou(SEQUENCE_IDX, pred_bbs))
    print(dataset.calculate_auc(SEQUENCE_IDX, pred_bbs))