from polimi.preprocessing_pipelines.pre_115f import build_features_iterator as build_features_iterator_115f
from polimi.preprocessing_pipelines.pre_94f import build_features_iterator as build_features_iterator_94f
from polimi.preprocessing_pipelines.pre_68f import build_features_iterator as build_features_iterator_68f
from polimi.preprocessing_pipelines.pre_127 import build_features_iterator as build_features_iterator_127f

from polimi.preprocessing_pipelines.pre_115f import build_features as build_features_115f
from polimi.preprocessing_pipelines.pre_94f import build_features as build_features_94f
from polimi.preprocessing_pipelines.pre_68f import build_features as build_features_68f
from polimi.preprocessing_pipelines.pre_127 import build_features as build_features_127f

from polimi.preprocessing_pipelines.pre_127 import build_features_iterator_test as build_features_iterator_test_127f

PREPROCESSING_VERSIONS = ['68f','94f','115f','127f','latest']

PREPROCESSING = {
    '68f': build_features_68f,
    '94f': build_features_94f,
    '115f': build_features_115f,
    '127f': build_features_127f,
    'latest': build_features_127f
}

BATCH_PREPROCESSING = {
    '68f': build_features_iterator_68f,
    '94f': build_features_iterator_94f,
    '115f': build_features_iterator_115f,
    '127f': build_features_iterator_127f,
    'latest': build_features_iterator_127f,
}

TEST_BATCH_PREPROCESSING = {
    '127f' : build_features_iterator_test_127f,
    'latest' : build_features_iterator_test_127f
}

def get_preprocessing_version(version):
    return PREPROCESSING[version]

def get_batch_preprocessing_version(version):
    return BATCH_PREPROCESSING[version]

def get_test_batch_preprocessing_version(version):
    return TEST_BATCH_PREPROCESSING[version]