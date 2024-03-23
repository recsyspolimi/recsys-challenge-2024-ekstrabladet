import polars as pl
import pathlib as Path

from src.ebrec.code.utils.constants import PATH


def load_history_behaviors(bundle = "large", split = "train", filename = "behaviors"):
    print("I'm loading {} from {}, bundle: {}".format(filename,split, bundle))
    print(PATH)
    return pl.read_parquet(Path(str(PATH) + bundle).joinpath(split,filename+".parquet"))


def load_articles(bundle = "large"):
    print("I'm loading articles bundle: {}".format(bundle))
    return pl.read_parquet(Path(str(PATH) + bundle).joinpath("articles.parquet"))