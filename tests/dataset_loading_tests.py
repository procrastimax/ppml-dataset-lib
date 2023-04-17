import unittest
from ppml_datasets import MnistDataset


class MnistTestCase(unittest.TestCase):

    def setUp(self):
        self.mnist: MnistDataset = MnistDataset(model_img_shape=(
            28, 28, 3), builds_ds_info=True, batch_size=32, preprocessing_func=None)

    def test_load(self):
        self.assertRaises(Exception,  self.mnist.load_dataset())
        self.assertRaises(Exception,  self.mnist.prepare_datasets())
