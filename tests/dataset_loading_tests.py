import unittest
from ppml_datasets import MnistDataset, FashionMnistDataset


class MnistTestCase(unittest.TestCase):

    def setUp(self):
        self.mnist = MnistDataset(model_img_shape=(
            28, 28, 3), builds_ds_info=False, batch_size=32, preprocessing_func=None)

    def test_load(self):
        self.assertRaises(Exception,  self.mnist.load_dataset())
        self.assertRaises(Exception,  self.mnist.build_ds_info())
        self.assertRaises(Exception,  self.mnist.prepare_datasets())
        self.assertIsNotNone(self.mnist.ds_info)
        print(f"mnist dataset info: {self.mnist.ds_info}")


class FmnistTestCase(unittest.TestCase):

    def setUp(self):
        self.fmnist = FashionMnistDataset(model_img_shape=(
            28, 28, 3), builds_ds_info=False, batch_size=32, preprocessing_func=None)

    def test_load(self):
        self.assertRaises(Exception,  self.fmnist.load_dataset())
        self.assertRaises(Exception,  self.fmnist.build_ds_info())
        self.assertRaises(Exception,  self.fmnist.prepare_datasets())
        self.assertIsNotNone(self.fmnist.ds_info)
        print(f"fmnist dataset info: {self.fmnist.ds_info}")
