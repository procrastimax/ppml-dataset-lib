import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from sklearn.utils import class_weight

from tensorflow.keras.layers import Layer, Resizing, Rescaling, RandomFlip, RandomRotation, RandomTranslation, RandomZoom
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Tuple, Dict, Any, Optional, List, Union, Callable
import scipy.stats
import os
import math

from ppml_datasets.utils import get_ds_as_numpy


@dataclass(eq=True, frozen=False)
class AbstractDataset():
    dataset_name: str
    dataset_path: Optional[str]
    # shape that the dataset should be transformed to
    model_img_shape: Tuple[int, int, int]

    batch_size: Optional[int]
    convert_to_rgb: bool
    augment_train: bool

    shuffle: bool
    is_tfds_ds: bool
    # if True, automatically builds ds_info after loading dataset data
    builds_ds_info: bool = field(default=False, repr=False)

    # model specific preprocessing for the dataset like: tf.keras.applications.resnet50.preprocess_input
    preprocessing_function: Optional[Callable[[float], tf.Tensor]] = None

    variants: Optional[List[Dict]] = None

    # image shape of the original dataset data (currently has no real function but interesting to know)
    dataset_img_shape: Optional[Tuple[int, int, int]] = None
    # optionally providable class_names, only for cosmetic purposes when printing out ds_info
    class_names: Optional[List[str]] = None

    random_rotation: Optional[float] = 0.1
    random_zoom: Optional[float] = 0.15
    random_flip: Optional[str] = "horizontal"
    random_brightness: Optional[float] = 0.1
    random_translation_width: Optional[float] = 0.1
    random_translation_height: Optional[float] = 0.1

    random_seed: int = 42
    repeat: bool = False

    class_labels: Optional[Tuple[Any]] = None
    class_counts: Optional[Tuple[int]] = None
    class_distribution: Optional[Tuple[int]] = None

    train_val_test_split: Tuple[float, float, float] = field(init=False)

    # TODO: make this a dataclass instead of dict -> since we need the attributes well defined through the code
    ds_info: Dict[str, Any] = field(init=False, default_factory=dict)
    ds_train: tf.data.Dataset = field(init=False, repr=False)
    ds_val: tf.data.Dataset = field(init=False, repr=False, default=None)
    ds_test: Optional[tf.data.Dataset] = field(init=False, repr=False, default=None)

    ds_attack_train: tf.data.Dataset = field(init=False, repr=False, default=None)
    ds_attack_test: tf.data.Dataset = field(init=False, repr=False, default=None)

    def _load_dataset(self):
        """Load dataset from tfds library.

        This function should be overwritten by all classes which do not utilize the tfds library to load the dataset.
        Overwrite this function with the needed functionality to load the dataset from files. Then call the 'load_dataset()' function to bundle
        data loading and dataset info creation.
        """
        if self.is_tfds_ds:
            self.__load_from_tfds()

    def load_dataset(self, fn_filter=None):
        print(f"Loading {self.dataset_name}")
        self._load_dataset()
        if self.builds_ds_info:
            self.build_ds_info()

        if fn_filter is not None:
            if self.ds_train is not None:
                self.ds_train = self.ds_train.filter(fn_filter)

                # we need to reset cardinality since is likely that the info is lost after filtering
                ds_len = sum(1 for _ in self.ds_train)
                self.ds_train = self.ds_train.apply(tf.data.experimental.assert_cardinality(ds_len))

            if self.ds_test is not None:
                self.ds_test = self.ds_test.filter(fn_filter)

                # we need to reset cardinality since is likely that the info is lost after filtering
                ds_len = sum(1 for _ in self.ds_test)
                self.ds_test = self.ds_test.apply(tf.data.experimental.assert_cardinality(ds_len))

            if self.ds_val is not None:
                self.ds_val = self.ds_val.filter(fn_filter)

                # we need to reset cardinality since is likely that the info is lost after filtering
                ds_len = sum(1 for _ in self.ds_val)
                self.ds_val = self.ds_val.apply(tf.data.experimental.assert_cardinality(ds_len))

    def __load_from_tfds(self):
        """Load dataset from tensorflow_datasets via 'dataset_name'."""
        if not self.is_tfds_ds:
            print("Cannot load dataset from tfds since it is not a tfds dataset!")
            return

        if self.dataset_path is not None:
            data_dir = os.path.join(self.dataset_path, self.dataset_name)
        else:
            data_dir = None

        ds_dict: dict = tfds.load(
            name=self.dataset_name,
            data_dir=data_dir,
            as_supervised=True,
            with_info=False
        )

        if "val" in ds_dict.keys():
            self.ds_val = ds_dict["val"]
            print("Loaded validation DS")
        if "test" in ds_dict.keys():
            self.ds_test = ds_dict["test"]
            print("Loaded test DS")
        if "train" in ds_dict.keys():
            self.ds_train = ds_dict["train"]
            print("Loaded train DS")

    def split_val_from_train(self, val_split: float = 0.3) -> Tuple[int, int]:
        """Split train dataset into validation and train dataset.

        Returns new length of train and validation DS
        """
        self.ds_train, self.ds_val = tf.keras.utils.split_dataset(
            self.ds_train, right_size=val_split,
            shuffle=True, seed=self.random_seed)

        return (len(self.ds_train), len(self.ds_val))

    def resplit_datasets(self, train_val_test_split: Tuple[float, float, float], percentage_loaded_data: int = 100):
        """Resplits all datasets (train, val, test) into new split values.

        First all current datasets are merged into one, than the datasets are resplitted into the specified split parts.
        If percentage_loaded_data is specified, than only this fraction of the merged dataset is used for splitting,
        effectively reducing the number of samples in each dataset.
        """
        ds = self.ds_train
        if self.ds_test is not None:
            ds = ds.concatenate(self.ds_test)

        if self.ds_val is not None:
            ds = ds.concatenate(self.ds_val)

        if percentage_loaded_data != 100:
            new_ds_size = math.ceil(len(ds) * (self.percentage_loaded_data / 100.0))
            ds = ds.take(new_ds_size)

        train_split = self.train_val_test_split[0]
        val_split = self.train_val_test_split[1]
        test_split = self.train_val_test_split[2]

        self.ds_train, right_ds = tf.keras.utils.split_dataset(
            ds, left_size=train_split,
            shuffle=True, seed=self.random_seed)

        if val_split == 0.0:
            self.ds_test = right_ds
        elif test_split == 0.0:
            self.ds_val = right_ds
        else:
            # shuffling once should be enough
            self.ds_val, self.ds_test = tf.keras.utils.split_dataset(
                right_ds, left_size=val_split / (val_split + test_split), shuffle=False)

    def set_augmentation_parameter(self, random_flip: Optional[str],
                                   random_rotation: Optional[float] = 0.1,
                                   random_zoom: Optional[float] = 0.15,
                                   random_brightness: Optional[float] = 0.1,
                                   random_translation_width: Optional[float] = 0.1,
                                   random_translation_height: Optional[float] = 0.1):
        self.random_flip = random_flip
        self.random_rotation = random_rotation
        self.random_zoom = random_zoom
        self.random_brightness = random_brightness
        self.random_translation_height = random_translation_height
        self.random_translation_width = random_translation_width

    def merge_all_datasets(self, percentage_loaded_data: int = 100) -> tf.data.Dataset:
        """Merge all datasets (train, val, test) into train dataset.

        A percentage can be specified, than only this percentage of the old data is used for the new train_ds after merging.
        """
        ds = self.ds_train
        if self.ds_test is not None:
            ds = ds.concatenate(self.ds_test)

        if self.ds_val is not None:
            ds = ds.concatenate(self.ds_val)

        if percentage_loaded_data != 100:
            new_ds_size = math.ceil(len(ds) * (self.percentage_loaded_data / 100.0))
            ds = ds.take(new_ds_size)

        self.ds_train = ds

    def set_class_names(self, class_names: List[str]):
        self.class_names = class_names

    def prepare_datasets(self):
        """Prepare all currently stored datasets (train, val, test) and the corresponding attack datsets (train, test).

        Preparation can include data shuffling, augmentation and resnet50-preprocessing.
        Augmentation is applied to train dataset if specified, augmentation is never applied to validation or test dataset

        """
        # prepare attack datasets
        # we need to first prepare the attack DS since they depend on the unmodified original datasets
        self.ds_attack_train = self.prepare_ds(self.ds_train, cache=True, resize_rescale=True,
                                               img_shape=self.model_img_shape,
                                               batch_size=1, convert_to_rgb=self.convert_to_rgb,
                                               preprocessing_func=self.preprocessing_function,
                                               shuffle=False, augment=False)
        if self.ds_test is not None:
            self.ds_attack_test = self.prepare_ds(self.ds_test, cache=True, resize_rescale=True,
                                                  img_shape=self.model_img_shape,
                                                  batch_size=1, convert_to_rgb=self.convert_to_rgb,
                                                  preprocessing_func=self.preprocessing_function,
                                                  shuffle=False, augment=False)

        self.ds_train = self.prepare_ds(self.ds_train, cache=True, resize_rescale=True,
                                        img_shape=self.model_img_shape,
                                        batch_size=self.batch_size, convert_to_rgb=self.convert_to_rgb,
                                        preprocessing_func=self.preprocessing_function,
                                        shuffle=self.shuffle, augment=self.augment_train)

        if self.ds_val is not None:
            self.ds_val = self.prepare_ds(self.ds_val, cache=True, resize_rescale=True,
                                          img_shape=self.model_img_shape,
                                          batch_size=self.batch_size,
                                          convert_to_rgb=self.convert_to_rgb,
                                          preprocessing_func=self.preprocessing_function,
                                          shuffle=False, augment=False)

        if self.ds_test is not None:
            self.ds_test = self.prepare_ds(self.ds_test, cache=True, resize_rescale=True,
                                           img_shape=self.model_img_shape, batch_size=self.batch_size,
                                           convert_to_rgb=self.convert_to_rgb,
                                           preprocessing_func=self.preprocessing_function,
                                           shuffle=False, augment=False)

    def prepare_ds(self, ds: tf.data.Dataset,
                   resize_rescale: bool,
                   img_shape: Tuple[int, int, int],
                   batch_size: Optional[int],
                   convert_to_rgb: bool,
                   preprocessing_func: Optional[Callable[[float], tf.Tensor]],
                   shuffle: bool,
                   augment: bool,
                   cache: Union[str, bool] = True) -> tf.data.Dataset:
        """Prepare datasets for training and validation for the ResNet50 model.

        This function applies image resizing, resnet50-preprocessing to the dataset. Optionally the data can be shuffled or further get augmented (random flipping, etc.)

        Parameter
        --------
        ds: tf.data.Dataset - dataset used for preparation steps
        resize_rescale: bool - if True, resizes the dataset to 'img_shape' and rescales all pixel values to a value between 0 and 255
        img_shape: Tuple[int, int, int] - if resize_rescale is True, than this value is used to rescale the image data to this size, consist of [height, width, color channel] -> only width and height are used for rescaling
        batch_size: int | None - batch size specified by integer value, if None is passed, no batching is applied to the data
        convert_to_rgb: bool - if True, the data is converted vom grayscale to rgb values
        preprocessing: bool - if True, model specific preprocessing is applied to the data (currently resnet50_preprocessing)
        shuffle: bool - if True, the data is shuffled, the used shuffle buffer for this has the size of the data
        augment: bool - if True, data augmentation (random flip, random rotation, random translation, random zoom, random brightness) is applied to the data

        """
        AUTOTUNE = tf.data.AUTOTUNE

        preprocessing_layers = tf.keras.models.Sequential()
        if convert_to_rgb:
            preprocessing_layers.add(GrayscaleToRgb())

        if resize_rescale:
            preprocessing_layers.add(Resizing(img_shape[0], img_shape[1]))
            preprocessing_layers.add(Rescaling(scale=1. / 255.))

        if preprocessing_func:
            preprocessing_layers.add(ModelPreprocessing(preprocessing_func))

        if convert_to_rgb or resize_rescale or preprocessing_func:
            ds = ds.map(lambda x, y: (preprocessing_layers(x), y),
                        num_parallel_calls=AUTOTUNE)

        if cache:
            if isinstance(cache, str):
                ds = ds.cache(cache)
            else:
                ds = ds.cache()

        if shuffle:
            ds = ds.shuffle(buffer_size=ds.cardinality().numpy(), seed=self.random_seed)

        if batch_size is not None:
            ds = ds.batch(batch_size, num_parallel_calls=AUTOTUNE)

        if augment:
            augmentation_layers = tf.keras.models.Sequential()

            if self.random_flip:
                augmentation_layers.add(RandomFlip(self.random_flip))

            if self.random_rotation:
                augmentation_layers.add(RandomRotation(self.random_rotation, fill_mode="constant"))

            if self.random_translation_width and self.random_translation_height:
                augmentation_layers.add(RandomTranslation(self.random_translation_height,
                                                          self.random_translation_width, fill_mode="constant"))
            if self.random_zoom:
                augmentation_layers.add(RandomZoom(self.random_zoom, fill_mode="constant"))

            if self.random_brightness:
                augmentation_layers.add(RandomBrightness(self.random_brightness))

            ds = ds.map(lambda x, y: (augmentation_layers(x, training=True), y),
                        num_parallel_calls=AUTOTUNE)

        # Use buffered prefetching on all datasets.
        return ds.prefetch(buffer_size=AUTOTUNE)

    def calculate_class_weights(self) -> Tuple[Optional[Dict[int, int]], Optional[Dict[int, float]]]:
        """Calculate class weights and class counts for train dataset."""
        class_labels, class_counts, class_distribution = self.get_class_distribution()

        class_counts_dict: Dict[str, int] = {}
        for y, count in zip(class_labels, class_counts):
            if self.class_names is not None and len(self.class_names) == len(class_labels):
                class_counts_dict[f"{self.class_names[y]}({y})"] = count
            else:
                class_counts_dict[y] = count

        weights = class_weight.compute_class_weight(class_weight='balanced',
                                                    classes=np.unique(class_distribution),
                                                    y=class_distribution)

        class_weights: Dict[str, float] = {}
        if self.class_names is not None and len(self.class_names) == len(class_labels):
            for i, weight in enumerate(weights):
                class_weights[f"{self.class_names[y]}({y})"] = weight
        else:
            class_weights = dict(enumerate(weights))
        return (class_counts_dict, class_weights)

    def get_dataset_count(self) -> Dict[str, int]:
        """Calculate number of datapoints for each part of the dataset (train,test,val)."""
        ds_count: Dict[str, int] = defaultdict(int)
        if self.ds_train is not None:
            ds_count["train"] = self.ds_train.cardinality().numpy()

        if self.ds_val is not None:
            ds_count["val"] = self.ds_val.cardinality().numpy()

        if self.ds_test is not None:
            ds_count["test"] = self.ds_test.cardinality().numpy()

        return ds_count

    def get_class_distribution(self, ds: Optional[tf.data.Dataset] = None, force_recalcuation: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate and return absolute class distribution from train dataset.

        This function returns the desired class_labels, class_counts and class_distribution values but also sets these variables as class variables.
        This is useful to not execute the function again, but only return the class variables, unless the 'force_recalcuation' flag is set to True.

        Parameter:
        --------
        ds: tf.data.Dataset - an optional dataset can be given to this function to calculate the class distribution of the given datasets
                              if ds is not set, it is assumed to calculate the class distribution from the current train dataset
        force_recalcuation: bool - (default = False), if set to True, this function calculates the class_distribution again

        Return:
        ------
        (np.ndarray, np.ndarray, np.ndarray): three numpy arrays
            -> first one containing the class number
            -> second one containing the number of datapoints in the class (ordered)
            -> third one as a class representation for all datapoints
        f.e.: ([1,2,3,4,5],[404,133,313,122,10], [4,1,0,2,5,4,1,4,3,2,4,3,3,1,...])

        """
        if self.class_counts is not None and self.class_labels is not None and self.class_distribution is not None and force_recalcuation is not True:
            return (self.class_labels, self.class_counts, self.class_distribution)

        if ds is not None:
            y_train = np.fromiter(ds.map(lambda _, y: y), int)
        else:
            y_train = np.fromiter(self.ds_train.map(lambda _, y: y), int)

        distribution = np.unique(y_train, return_counts=True)

        self.class_labels = distribution[0]
        self.class_counts = distribution[1]
        self.class_distribution = y_train

        return distribution + (y_train,)

    def calculate_class_imbalance(self) -> float:
        """Calculate class imbalance value for the train dataset.

        Idea of using shannon entropy to calculate balance from here: https://stats.stackexchange.com/questions/239973/a-general-measure-of-data-set-imbalance
        For data of n instances, if we have k classes of size c_i, we can calculate the entropy

        Return:
        ------
        float:  class imbalance value [0,1]
                0 - unbalanced dataset
                1 - balanced dataset

        """
        _, class_counts, _ = self.get_class_distribution()

        n: int = sum(class_counts)
        k: int = len(class_counts)
        H: float = 0.0
        for c in class_counts:
            H += (c / n) * np.log((c / n))

        H *= -1
        B: float = H / np.log(k)
        return B

    def calculate_data_entropy(self, ds: Optional[tf.data.Dataset] = None) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Calculate and return data entropy values and normed entropy values.

        Parameter:
        --------
        ds: tf.data.Dataset - Optional, if passed, the entropy of the given dataset is calculated instead of the current training dataset

        Return:
        ------
        1 (float, float, float) : average entropy value, min entropy value, max entropy value
        2 (float, float, float) : normed average entropy value, normed min entropy value, normed max entropy value

        """
        entropy_val_list: List[float] = []
        normed_entropy_val_list: List[float] = []

        if ds is None:
            ds = self.ds_train

        for _, (data, _) in enumerate(ds):
            values, counts = np.unique(data, return_counts=True)
            entropy_val = scipy.stats.entropy(counts)
            entropy_val_list.append(entropy_val)

            normed_entropy_val = entropy_val / np.log(len(values))
            normed_entropy_val_list.append(normed_entropy_val)

        max_entropy = max(entropy_val_list)
        min_entropy = min(entropy_val_list)
        avg_entropy = sum(entropy_val_list) / len(entropy_val_list)

        normed_max_entropy = max(normed_entropy_val_list)
        normed_min_entropy = min(normed_entropy_val_list)
        normed_avg_entropy = sum(normed_entropy_val_list) / len(normed_entropy_val_list)

        return ((avg_entropy, min_entropy, max_entropy), (normed_avg_entropy, normed_min_entropy, normed_max_entropy))

    def build_ds_info(self):
        """Build dataset info dictionary.

        This function needs to be called after initializing and loading the dataset
        """
        class_counts, class_weights = self.calculate_class_weights()
        ds_count = self.get_dataset_count()
        total_count: int = sum(ds_count.values())
        class_imbalance: float = self.calculate_class_imbalance()
        entropy_values, normed_entropy_values = self.calculate_data_entropy()

        self.ds_info = {
            'name': self.dataset_name,
            'dataset_img_shape': self.dataset_img_shape,
            'model_img_shape': self.model_img_shape,
            'total_count': total_count,
            'train_count': ds_count["train"],
            'val_count': ds_count["val"],
            'test_count': ds_count["test"],
            'classes': len(class_counts),
            'class_imbalance': class_imbalance,
            'class_counts': class_counts,
            'class_weights': class_weights,
            'avg_entropy': entropy_values[0],
            'min_entropy': entropy_values[1],
            'max_entropy': entropy_values[2],
            'normed_avg_entropy': normed_entropy_values[0],
            'normed_min_entropy': normed_entropy_values[1],
            'normed_max_entropy': normed_entropy_values[2],
        }

    def get_train_ds_as_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return Train Dataset as unbatched (values, labels) numpy arrays."""
        return get_ds_as_numpy(self.ds_train)

    def get_test_ds_as_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return Test Dataset as unbatched (values, labels) numpy arrays."""
        return get_ds_as_numpy(self.ds_test)

    def get_val_ds_as_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return Validation Dataset as unbatched (values, labels) numpy arrays."""
        return get_ds_as_numpy(self.ds_val)

    def get_attack_train_ds_as_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return Attack Train Dataset as unbatched (values, labels) numpy arrays."""
        return get_ds_as_numpy(self.ds_attack_train)

    def get_attack_test_ds_as_numpy(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return Attack Test Dataset as unbatched (values, labels) numpy arrays."""
        return get_ds_as_numpy(self.ds_attack_test)


class GrayscaleToRgb(Layer):
    """Layer for converting 1-channel grayscale input to 3-channel rgb."""

    def __init__(self, **kwargs):
        """Initialize GrayscaleToRgb layer."""
        super().__init__(**kwargs)

    def call(self, x):
        return tf.image.grayscale_to_rgb(x)


class RandomBrightness(Layer):
    """Layer for random brightness augmentation in images."""

    def __init__(self, factor=0.1, **kwargs):
        """Initialize RandomBrightness layer."""
        super().__init__(**kwargs)
        self.factor = factor

    def call(self, x):
        return tf.image.random_brightness(x, max_delta=self.factor)


class ModelPreprocessing(Layer):
    """Layer for specific model preprocessing steps."""

    def __init__(self, pre_func: Callable[[float], tf.Tensor], **kwargs):
        """Initialize layer for model preprocessing."""
        super().__init__(**kwargs)
        self.pre_func = pre_func

    def call(self, x):
        return self.pre_func(x)
