"""
配置文件，所有可配置的变量都集中于此，并提供默认值
"""


import os

# BASE_DIR = '/Users/tsw/ScenicSpotReviews'

# W2V_DIR = BASE_DIR + '/embeddings/'

# TEXT_DATA_DIR = BASE_DIR + '/data/'

# MAX_SEQUENCE_LENGTH = 100





BATCH_SIZE = 32


# C
CLASS_NUM = 3


# D
DATA_DIR = "./data/training-inspur.csv"
DATA_DIR = os.path.abspath(DATA_DIR)


# E
EMBEDDING_DIM = 300

# M
MAX_SEQUENCE_LENGTH = 150
MAX_NUM_WORDS = 33950
MAX_NB_WORDS = 30000

# V
VALIDATION_SPLIT = 0.2

# W
W2V_FILE = "./embeddings/zhihu.vec"
WORD_DICT = "/Users/burness/git_repository/dl_opensource/nlp/oxford-cs-deepnlp-2017/practical-2/data/origin_data/t_tag_infos.txt"


