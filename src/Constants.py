# This file as all the constants for the project
from os.path import expanduser

# Paths
BASE_DIR = expanduser('~/nnClassify/data')
MODELS_DIR = BASE_DIR + 'Models/'
INCEPTION_DIR = BASE_DIR + 'inception/'
TRAIN_DIR = BASE_DIR +  'Train/'
PRED_DIR = BASE_DIR + 'Predictions/'
TEST_DIR = BASE_DIR + 'Test/'

# Categories
ABNORMAL = "Abnormal"
NORMAL = "Normal"
WEIRD = "Weird"
NOT_SURE = "Not_Sure"

# Files
RESULTS = "results.txt"