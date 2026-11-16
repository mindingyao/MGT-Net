import os
import sys
import tempfile

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import transform
from dataset import Dataset


class _Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        for transform_fn in self.transforms:
            sample = transform_fn(sample)
        return sample


def test_duts_tr_flow_matches_image():
    """DUTS-TR has no optical flow, so its zero flow must match the resized image."""
    transforms = _Compose([
        transform.FixedResize(size=(448, 448)),
        transform.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        transform.ToTensor(),
    ])

    with tempfile.TemporaryDirectory() as root:
        data_dir = os.path.join(root, 'dataset', 'train', 'DUTS-TR')
        os.makedirs(data_dir)
        Image.fromarray(np.zeros((300, 400, 3), dtype=np.uint8)).save(
            os.path.join(data_dir, 'img.png'))
        Image.fromarray(np.zeros((300, 400), dtype=np.uint8)).save(
            os.path.join(data_dir, 'gt.png'))
        with open(os.path.join(data_dir, 'train.txt'), 'w') as imgset_file:
            imgset_file.write('dataset/train/DUTS-TR/img.png dataset/train/DUTS-TR/gt.png\n')

        cwd = os.getcwd()
        os.chdir(root)
        try:
            sample = Dataset(datasets=['DUTS-TR'], transform=transforms, mode='train')[0]
        finally:
            os.chdir(cwd)

    assert tuple(sample['image'].shape) == (3, 448, 448)
    assert tuple(sample['flow'].shape) == tuple(sample['image'].shape)
    assert not sample['flow'].any()


if __name__ == '__main__':
    test_duts_tr_flow_matches_image()
    print('PASSED')
