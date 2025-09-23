import os
from collections import defaultdict
from typing import Dict, List
from pprint import pprint
import random
from PIL import Image
import shutil

IMAGES_BASE_PATH = os.path.join(os.getcwd(), "archive", "asl_dataset")
IMAGES_SPLIT_PATH = os.path.join(os.getcwd(), "dataset")
IMAGES_PER_CLASS = 70
TRAIN_SET_SIZE = int(IMAGES_PER_CLASS * 0.8)
TEST_SET_SIZE = int(IMAGES_PER_CLASS * 0.1)
VAL_SET_SIZE = int(IMAGES_PER_CLASS * 0.1)
class SetupImages:
    def __init__(self):
        self.labels_to_paths: Dict[str, List] = defaultdict(list)
        self._train: Dict[str, List] = defaultdict(list)
        self._test: Dict[str, List] = defaultdict(list)
        self._val: Dict[str, List] = defaultdict(list)
        self.datasets = {
            "train": self._train,
            "test": self._test,
            "val": self._val
        }

        self._create_dataset_object()

    def _create_dataset_object(self) -> None:
        """samples from the dataset to create train, test, and val datasets
        """
        print("opening images")
        for dir in os.listdir(IMAGES_BASE_PATH):
            if os.path.isdir(os.path.join(IMAGES_BASE_PATH, dir)):
                for file in os.listdir(os.path.join(IMAGES_BASE_PATH, dir)):
                    self.labels_to_paths[dir].append(os.path.join(IMAGES_BASE_PATH, dir, file))

    def create_data_splits(self):
        print("splitting into train, test, and val")
        for key in self.labels_to_paths.keys():
            random.shuffle(self.labels_to_paths[key])
            self._train[key] = self.labels_to_paths[key][:TRAIN_SET_SIZE]
            self._test[key] = self.labels_to_paths[key][TRAIN_SET_SIZE:TRAIN_SET_SIZE + TEST_SET_SIZE]
            self._val[key] = self.labels_to_paths[key][-VAL_SET_SIZE:]
    
    def move_files(self, dataset: str):
        for key, value in self.datasets[dataset].items():
            classPath = os.path.join(IMAGES_SPLIT_PATH, dataset, key)
            if not os.path.exists(classPath):
                os.mkdir(classPath)
            for file in value:
                shutil.copyfile(file, os.path.join(classPath, os.path.basename(file)))

    def create_train_test_val_sets(self):
        self.move_files("train")
        self.move_files("test")
        self.move_files("val")


    def get_train(self):
        return self._train

    def get_test(self):
        return self._test
    
    def get_val(self):
        return self._val

if __name__ == "__main__":
    setupImages = SetupImages()
    setupImages.create_data_splits()
    setupImages.create_train_test_val_sets()