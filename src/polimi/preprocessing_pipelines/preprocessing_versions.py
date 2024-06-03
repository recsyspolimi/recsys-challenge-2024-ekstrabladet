from polimi.preprocessing_pipelines.pre_115f import build_features_iterator as build_features_iterator_115f
from polimi.preprocessing_pipelines.pre_94f import build_features_iterator as build_features_iterator_94f
from polimi.preprocessing_pipelines.pre_68f import build_features_iterator as build_features_iterator_68f
from polimi.preprocessing_pipelines.pre_127 import build_features_iterator as build_features_iterator_127f
from polimi.preprocessing_pipelines.pre_142 import build_features_iterator as build_features_iterator_142f
from polimi.preprocessing_pipelines.pre_147 import build_features_iterator as build_features_iterator_147f
from polimi.preprocessing_pipelines.pre_new import build_features_iterator as build_features_iterator_new
from polimi.preprocessing_pipelines.pre_new2_click_urms import build_features_iterator as build_features_iterator_new_click
from polimi.preprocessing_pipelines.pre_new_with_recsys import build_features_iterator as build_features_iterator_new_with_recsys


from polimi.preprocessing_pipelines.pre_115f import build_features as build_features_115f
from polimi.preprocessing_pipelines.pre_94f import build_features as build_features_94f
from polimi.preprocessing_pipelines.pre_68f import build_features as build_features_68f
from polimi.preprocessing_pipelines.pre_127 import build_features as build_features_127f
from polimi.preprocessing_pipelines.pre_142 import build_features as build_features_142f
from polimi.preprocessing_pipelines.pre_147 import build_features as build_features_147f
from polimi.preprocessing_pipelines.pre_new import build_features as build_features_new
from polimi.preprocessing_pipelines.pre_new2_click_urms import build_features as build_features_new_click
from polimi.preprocessing_pipelines.pre_new_emb_urm import build_features as build_features_new_emb_urm


from polimi.preprocessing_pipelines.pre_127 import build_features_iterator_test as build_features_iterator_test_127f
from polimi.preprocessing_pipelines.pre_142 import build_features_iterator_test as build_features_iterator_test_142f
from polimi.preprocessing_pipelines.pre_147 import build_features_iterator_test as build_features_iterator_test_147f
from polimi.preprocessing_pipelines.pre_new import build_features_iterator_test as build_features_iterator_test_new
from polimi.preprocessing_pipelines.pre_new2_click_urms import build_features as build_features_iterator_test_new_click
from polimi.preprocessing_pipelines.pre_new_with_recsys import build_features_iterator_test as build_features_iterator_test_new_with_recsys




PREPROCESSING = {
    '68f': build_features_68f,
    '94f': build_features_94f,
    '115f': build_features_115f,
    '127f': build_features_127f,
    '142f': build_features_142f,
    '147f': build_features_147f,
    'new': build_features_new,
    'new_click': build_features_new_click,
    'new_emb_urm': build_features_new_emb_urm,
    'latest': build_features_127f
}
PREPROCESSING_VERSIONS = list(PREPROCESSING.keys())


BATCH_PREPROCESSING = {
    '68f': build_features_iterator_68f,
    '94f': build_features_iterator_94f,
    '115f': build_features_iterator_115f,
    '127f': build_features_iterator_127f,
    '142f': build_features_iterator_142f,
    '147f': build_features_iterator_147f,
    'new': build_features_iterator_new,
    'new_click' : build_features_iterator_new_click,
    'new_with_recsys': build_features_iterator_new_with_recsys,
    'latest' : build_features_iterator_127f,
}

TEST_BATCH_PREPROCESSING = {
    '127f' : build_features_iterator_test_127f,
    '142f': build_features_iterator_test_142f,
    '147f': build_features_iterator_test_147f,
    'new': build_features_iterator_test_new,
    'new_click': build_features_iterator_test_new_click,
    'latest' : build_features_iterator_test_127f,
    'new_with_recsys': build_features_iterator_new_with_recsys,
}

def get_preprocessing_version(version):
    return PREPROCESSING[version]

def get_batch_preprocessing_version(version):
    return BATCH_PREPROCESSING[version]

def get_test_batch_preprocessing_version(version):
    return TEST_BATCH_PREPROCESSING[version]