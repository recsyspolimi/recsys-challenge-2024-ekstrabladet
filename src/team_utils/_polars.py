import polars as pl 
from rich.progress import Progress
from typing import Callable, List, Any


def list_pct_matches_with_col(a: str, b: str) -> pl.Expr:
    '''
    Returns an expression to count the number of matching element in a list with another column.
    The polars function count_matches cannot be used since it wants only a single element, 
    variable element from row to row.
    
    Args:
        a: the column containing lists, must be of type pl.List(any) (e.g. pl.List(pl.Int8))
        b: the column to match to the list, must be of the same type contained in the lists of a (e.g. pl.Int8)
        
    Returns:
        a pl.Expr that can be used to count the percentage of matching elements of b inside a
        
    Example:
    >>> x = pl.DataFrame({'col1': [[1, 2, 2], [3]], 'col2': [2, 4]})
    >>> x.with_columns(list_pct_matches_with_col('col1', 'col2').alias('pct_matches'))
    shape: (2, 3)
    ┌───────────┬──────┬──────────────┐
    │ col1      ┆ col2 ┆ pct_matches  ┆
    │ ---       ┆ ---  ┆ ---          │
    │ list[i64] ┆ i64  ┆ f64          │
    ╞═══════════╪══════╪══════════════╡
    │ [1, 2, 2] ┆ 2    ┆ 0.33         │
    │ [3]       ┆ 4    ┆ 0.0          │
    └───────────┴──────┴──────────────┘
    '''
    return pl.when(pl.col(a).list.len() == 0).then(0.0) \
        .otherwise((pl.col(a).list.len() - (pl.col(a).list.set_difference(pl.col(b))).list.len()) / pl.col(a).list.len())


def get_single_feature_function(df: pl.DataFrame, index_feature: str, f_name: str, 
                                progress_bar: Progress, progress_task) -> Callable[[List[Any]], List[Any]]:
    '''
    Returns a function to be used to get a list of elements from a column of the specified dataframe. This column can have
    any type. For example if the column contains a dtype pl.List(pl.Int32) the output will be a List[List[int]].
    It can be used together with .map_elements(...) applied to a list column of a dataframe.
    
    Args:
        df: the dataframe from which to extract the list of values
        index_feature: the column that will be used to filter the dataframe with the specified list when calling the function
        f_name: the name of the feature to be extracted
        progress_bar: a progress bar to be shown during the process
        progress_task: the task connected to the progress bar
        
    Returns:
        a function taking a list of values as input
        
    Example:
    >>> x = pl.DataFrame({'col1': [[1, 2, 2], [3], [4]], 'col2': [2, 4, 4]})
    >>> x
    shape: (3, 2)
    ┌───────────┬──────┐
    │ col1      ┆ col2 ┆
    │ ---       ┆ ---  │
    │ list[i64] ┆ i64  │
    ╞═══════════╪══════╡
    │ [1, 2, 2] ┆ 2    │
    │ [3]       ┆ 4    │
    │ [4]       ┆ 4    │
    └───────────┴──────┘
    >>> y = pl.DataFrame({'x': [2, 4]})
    >>> bar = Progress()
    >>> task = progress.add_task("Fetching col1", total=y.shape[0])
    >>> y.with_columns(pl.col('x').map_elements(get_single_feature_function(x, 'col2', 'col1', bar, task), return_dtype=pl.List(pl.Int64)).alias('col1_values'))
    shape: (2, 3)
    ┌─────┬───────────┐
    │ x   ┆ col1      │
    │ --- ┆ ---       │
    │ i64 ┆ list[i64] │
    ╞═════╪═══════════╡
    │ 2   ┆ [1, 2, 2] │
    │ 4   ┆ [3, 4]    │
    └─────┴───────────┘
    '''
    def get_feature(article_ids):
        progress_bar.update(progress_task, advance=1)
        feature_values = df.filter(pl.col(index_feature).is_in(article_ids)) \
            .select(pl.col(f_name))[f_name].to_list()
        return feature_values
    return get_feature


def get_unique_list_feature_function(df: pl.DataFrame, index_feature: str, f_name: str, 
                                     progress_bar: Progress, progress_task) -> Callable[[List[Any]], List[Any]]:
    '''
    Returns a function to be used to get a list of elements from a column of the specified dataframe. This column must have a dtype
    of type pl.List(...). After filtering the elements by the specified list of ids (referring to the index_feature), 
    the function will first computes the unique elements inside the list of every row, then it will concatenated all the lists in 
    another list so the output will be a list of lists.
    For example if the column contains a dtype pl.List(pl.Int32) the output will be a List[List[int]].
    It can be used together with .map_elements(...) applied to a list column of a dataframe.
    
    Args:
        df: the dataframe from which to extract the list of values
        index_feature: the column that will be used to filter the dataframe with the specified list when calling the function
        f_name: the name of the feature to be extracted
        progress_bar: a progress bar to be shown during the process
        progress_task: the task connected to the progress bar
        
    Returns:
        a function taking a list of values as input
        
    Example:
    >>> x = pl.DataFrame({'col1': [[1, 2, 2], [3], [4]], 'col2': [2, 4, 4]})
    >>> x
    shape: (3, 2)
    ┌───────────┬──────┐
    │ col1      ┆ col2 ┆
    │ ---       ┆ ---  │
    │ list[i64] ┆ i64  │
    ╞═══════════╪══════╡
    │ [1, 2, 2] ┆ 2    │
    │ [3]       ┆ 4    │
    │ [4]       ┆ 4    │
    └───────────┴──────┘
    >>> y = pl.DataFrame({'x': [2, 4]})
    >>> bar = Progress()
    >>> task = progress.add_task("Fetching col1", total=y.shape[0])
    >>> y.with_columns(pl.col('x').map_elements(get_unique_list_feature_function(x, 'col2', 'col1', bar, task), return_dtype=pl.List(pl.Int64)).alias('col1_values'))
    shape: (2, 3)
    ┌─────┬───────────┐
    │ x   ┆ col1      │
    │ --- ┆ ---       │
    │ i64 ┆ list[i64] │
    ╞═════╪═══════════╡
    │ 2   ┆ [1, 2]    │
    │ 4   ┆ [3, 4]    │
    └─────┴───────────┘
    '''
    def get_feature(article_ids):
        progress_bar.update(progress_task, advance=1)
        feature_values = df.filter(pl.col(index_feature).is_in(article_ids)) \
            .select(pl.col(f_name).list.unique())[f_name].to_list()
        return feature_values
    return get_feature


def get_unique_list_exploded_feature_function(df: pl.DataFrame, index_feature: str, f_name: str, 
                                              progress_bar: Progress, progress_task) -> Callable[[List[Any]], List[Any]]:
    '''
    Returns a function to be used to get a list of elements from a column of the specified dataframe. This column must have a dtype
    of type pl.List(...). After filtering the elements by the specified list of ids (referring to the index_feature), 
    the function will first computes the unique elements inside the list of every row, then it will concatenated all the lists and 
    flatten the output, so the final output is a single List.
    For example if the column contains a dtype pl.List(pl.Int32) the output will be a List[int].
    It can be used together with .map_elements(...) applied to a list column of a dataframe.
    
    Args:
        df: the dataframe from which to extract the list of values
        index_feature: the column that will be used to filter the dataframe with the specified list when calling the function
        f_name: the name of the feature to be extracted
        progress_bar: a progress bar to be shown during the process
        progress_task: the task connected to the progress bar
        
    Returns:
        a function taking a list of values as input
        
    Example:
    >>> x = pl.DataFrame({'col1': [[1, 2, 2], [3], [4]], 'col2': [2, 4, 4]})
    >>> x
    shape: (3, 2)
    ┌───────────┬──────┐
    │ col1      ┆ col2 ┆
    │ ---       ┆ ---  │
    │ list[i64] ┆ i64  │
    ╞═══════════╪══════╡
    │ [1, 2, 2] ┆ 2    │
    │ [3]       ┆ 4    │
    │ [4]       ┆ 4    │
    └───────────┴──────┘
    >>> y = pl.DataFrame({'x': [2, 4]})
    >>> bar = Progress()
    >>> task = progress.add_task("Fetching col1", total=y.shape[0])
    >>> y.with_columns(pl.col('x').map_elements(get_unique_list_feature_function(x, 'col2', 'col1', bar, task), return_dtype=pl.List(pl.Int64)).alias('col1_values'))
    shape: (2, 3)
    ┌─────┬─────────────────┐
    │ x   ┆ col1            │
    │ --- ┆ ---             │
    │ i64 ┆ list[list[i64]] │
    ╞═════╪═════════════════╡
    │ 2   ┆ [[1, 2]]        │
    │ 4   ┆ [[3], [4]]      │
    └─────┴─────────────────┘
    '''
    def get_feature(article_ids):
        progress_bar.update(progress_task, advance=1)
        feature_values = df.filter(pl.col(index_feature).is_in(article_ids)) \
            .select(pl.col(f_name).list.unique()).explode(f_name)[f_name].to_list()
        return feature_values
    return get_feature