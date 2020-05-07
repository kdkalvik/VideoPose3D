# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
from glob import glob
import os
import tqdm
import argparse
from pose_tracker import pose_tracker

def decode(filename):
    data = np.load(filename, allow_pickle=True)
    metadata =  {'w': 1920, 'h': 1080,}
    keypoints = data['keypoints']
    tracker = pose_tracker(data_frame=len(keypoints), num_joint=17)
    for i, frame in enumerate(keypoints):
        if len(frame.shape) != 3:
            print('file has empty skeleton_sequence')
            return
        tracker.update(frame, i+1)
    keypoints = tracker.get_skeleton_sequence()[:, :, :, :2]
    return keypoints, metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Custom dataset creator')
    parser.add_argument('-i', '--input',
                        required=True ,
                        type=str,
                        default='',
                        metavar='PATH',
                        help='detections directory')
    parser.add_argument('-o', '--output',
                        required=True,
                        type=str,
                        default='',
                        metavar='PATH',
                        help='output filename for 2D detections')
    args = parser.parse_args()

    metadata = {
        'layout_name': 'coco',
        'num_joints': 17,
        'keypoints_symmetry': [
            [1, 3, 5, 7, 9, 11, 13, 15],
            [2, 4, 6, 8, 10, 12, 14, 16],
        ],
        'video_metadata': {},
    }

    output = {}
    file_list = glob(args.input + '/*.npz')
    file_list = [f for f in file_list]
    for f in tqdm.tqdm(file_list):
        canonical_name = os.path.splitext(os.path.basename(f))[0]
        keypoints, video_metadata = decode(f)
        output[canonical_name] = {}
        output[canonical_name]['custom'] = [keypoints]
        metadata['video_metadata'][canonical_name] = video_metadata

    print('Saving...')
    np.savez_compressed(args.output, positions_2d=output, metadata=metadata)
    print('Done.')
