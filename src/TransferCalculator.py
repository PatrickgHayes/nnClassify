# This file is for calculating the output of the inception
# model so we can we that as input to our neural netwroks. 
# This is used for transfer learning. For more information
# see (https://github.com/Hvass-Labs/TensorFlow-Tutorials
#                   /blob/master/08_Transfer_Learning.ipynb)

import tensorflow as tf
import numpy as np
import os
from Constants import BASE_DIR
from Constants import TEST_DIR
from Constants import INCEPTION_DIR
from MyUtils_ import MyUtils
import pickle
import matplotlib.pyplot as plt
# Functions and classes for loading and using the Inception model.
import inception
from inception import transfer_values_cache


def getTransferValues(source_p, dest_p):
    inception.data_dir = INCEPTION_DIR

    inception.maybe_download()

    model = inception.Inception()

    file_path_cache_train = dest_p

    print("Processing Inception transfer-values ...")

    images = pickle.load( open(source_p, 'rb'))

    transfer_values_train = transfer_values_cache(
                                    cache_path=file_path_cache_train,
                                    images=images,
                                    model=model)
    return

def get_all_transfer_values(test_set):
    categories = MyUtils.listdir_nohidden(TEST_DIR + test_set + "/Images/")
    # The highest level directory will have a folder for each category
    # (0_Eyes, 1_Eye, 2_Eyes, etc.)
    for category in categories:
        os.mkdir(TEST_DIR + test_set + "/Transfer_Values/" + category)
        wells = list(MyUtils.listdir_nohidden(TEST_DIR + test_set + "/Pickles/"
                + category))
	
        for well in wells:
            getTransferValues(TEST_DIR + test_set + "/Pickles/"
                                + category + "/" + well,
                                TEST_DIR + test_set + "/Transfer_Values/"
                                + category + "/" + well)


