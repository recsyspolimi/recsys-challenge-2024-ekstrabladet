import polars as pl 
from rich.progress import Progress
from typing import Callable, List, Any
import numpy as np
import gc
from pathlib import Path
import polars.selectors as cs

def reduce_polars_df_memory_size(df: pl.DataFrame, verbose: bool = True) -> pl.DataFrame:
    start_mem = df.estimated_size('mb')
    if verbose:
        print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        # Integer types
        if col_type in [pl.Int16, pl.Int32, pl.Int64]:
            c_min = df[col].fill_null(0).min()
            c_max = df[col].fill_null(0).max()
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df = df.with_columns(pl.col(col).cast(pl.Int8))
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df = df.with_columns(pl.col(col).cast(pl.Int16))
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df = df.with_columns(pl.col(col).cast(pl.Int32))
        elif col_type in [pl.UInt16, pl.UInt32, pl.UInt64]:
            c_min = df[col].fill_null(0).min()
            c_max = df[col].fill_null(0).max()
            if c_min > np.iinfo(np.uint8).min and c_max < np.iinfo(np.uint8).max:
                df = df.with_columns(pl.col(col).cast(pl.UInt8))
            elif c_min > np.iinfo(np.uint16).min and c_max < np.iinfo(np.uint16).max:
                df = df.with_columns(pl.col(col).cast(pl.UInt16))
            elif c_min > np.iinfo(np.uint32).min and c_max < np.iinfo(np.uint32).max:
                df = df.with_columns(pl.col(col).cast(pl.UInt32))
        # Float types
        elif col_type == pl.Float64:
            c_min = df[col].fill_null(0).min()
            c_max = df[col].fill_null(0).max()
            if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                df = df.with_columns(pl.col(col).cast(pl.Float32))
        # List types
        elif col_type in [pl.List(pl.Int16), pl.List(pl.Int32), pl.List(pl.Int64)]:
            c_min = df[col].list.min().min()
            c_max = df[col].list.max().max()
            if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                df = df.with_columns(pl.col(col).cast(pl.List(pl.Int8)))
            elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                df = df.with_columns(pl.col(col).cast(pl.List(pl.Int16)))
            elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                df = df.with_columns(pl.col(col).cast(pl.List(pl.Int32)))
        elif col_type in [pl.List(pl.UInt16), pl.List(pl.UInt32), pl.List(pl.UInt64)]:
            c_min = df[col].list.min().min()
            c_max = df[col].list.max().max()
            if c_min > np.iinfo(np.uint8).min and c_max < np.iinfo(np.uint8).max:
                df = df.with_columns(pl.col(col).cast(pl.List(pl.UInt8)))
            elif c_min > np.iinfo(np.uint16).min and c_max < np.iinfo(np.uint16).max:
                df = df.with_columns(pl.col(col).cast(pl.List(pl.UInt16)))
            elif c_min > np.iinfo(np.uint32).min and c_max < np.iinfo(np.uint32).max:
                df = df.with_columns(pl.col(col).cast(pl.List(pl.UInt32)))

    gc.collect()
    end_mem = df.estimated_size('mb')
    if verbose:
        print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
        print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


def inflate_polars_df(df: pl.DataFrame, verbose: bool = True) -> pl.DataFrame:
    start_mem = df.estimated_size('mb')
    if verbose:
        print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype
        # Integer types
        if col_type in [pl.Int8, pl.Int16, pl.Int32, pl.Int64]:
            df = df.with_columns(pl.col(col).cast(pl.Int32))
        elif col_type in [pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64]:
            df = df.with_columns(pl.col(col).cast(pl.UInt32))
        # Float types
        elif col_type == pl.Float64:
            df = df.with_columns(pl.col(col).cast(pl.Float32))
        # List types
        elif col_type in [pl.List(pl.Int16), pl.List(pl.Int32), pl.List(pl.Int64)]:
            df = df.with_columns(pl.col(col).cast(pl.List(pl.Int32)))
        elif col_type in [pl.List(pl.UInt16), pl.List(pl.UInt32), pl.List(pl.UInt64)]:
            df.with_columns(pl.col(col).cast(pl.List(pl.UInt32)))
            
    if verbose:
        print('Memory usage of dataframe after cast is {:.2f} MB'.format(start_mem))
    return df


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
    ┌───────────┬──────┬─────────────┐
    │ col1      ┆ col2 ┆ pct_matches │
    │ ---       ┆ ---  ┆ ---         │
    │ list[i64] ┆ i64  ┆ f64         │
    ╞═══════════╪══════╪═════════════╡
    │ [1, 2, 2] ┆ 2    ┆ 0.666667    │
    │ [3]       ┆ 4    ┆ 0.0         │
    └───────────┴──────┴─────────────┘
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
    >>> task = bar.add_task("Fetching col1", total=y.shape[0])
    >>> y.with_columns(pl.col('x').map_elements(get_single_feature_function(x, 'col2', 'col1', bar, task), return_dtype=pl.List(pl.List(pl.Int64))).alias('col1_values'))
    shape: (2, 2)
    ┌─────┬─────────────────┐
    │ x   ┆ col1_values     │
    │ --- ┆ ---             │
    │ i64 ┆ list[list[i64]] │
    ╞═════╪═════════════════╡
    │ 2   ┆ [[1, 2, 2]]     │
    │ 4   ┆ [[3], [4]]      │
    └─────┴─────────────────┘
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
    >>> task = bar.add_task("Fetching col1", total=y.shape[0])
    >>> y.with_columns(pl.col('x').map_elements(get_unique_list_feature_function(x, 'col2', 'col1', bar, task), return_dtype=pl.List(pl.List(pl.Int64))).alias('col1_values'))
    shape: (2, 2)
    ┌─────┬─────────────────┐
    │ x   ┆ col1_values     │
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
    >>> task = bar.add_task("Fetching col1", total=y.shape[0])
    >>> y.with_columns(pl.col('x').map_elements(get_unique_list_exploded_feature_function(x, 'col2', 'col1', bar, task), return_dtype=pl.List(pl.Int64)).alias('col1_values'))
    shape: (2, 3)
    ┌─────┬─────────────┐
    │ x   ┆ col1_values │
    │ --- ┆ ---         │
    │ i64 ┆ list[i64]   │
    ╞═════╪═════════════╡
    │ 2   ┆ [1, 2]      │
    │ 4   ┆ [3, 4]      │
    └─────┴─────────────┘
    '''
    def get_feature(article_ids):
        progress_bar.update(progress_task, advance=1)
        feature_values = df.filter(pl.col(index_feature).is_in(article_ids)) \
            .select(pl.col(f_name).list.unique()).explode(f_name)[f_name].to_list()
        return feature_values
    return get_feature

def list_pct_matches_with_constant(a, value) -> pl.Expr:
    '''
    Returns an expression to count the percentage of matching element in a list with a constant value.
    The polars function count_matches cannot be used since it wants only a single element, 
    variable element from row to row.
    '''
    return pl.when(pl.col(a).list.len() == 0).then(0.0) \
        .otherwise(pl.col(a).list.count_matches(value) / pl.col(a).list.len())
        

def _convert_to_datetime(original_df_path, modified_df_path):
    '''
    Utility function used to convert the impression_time column to datetime in the modified dataframe.
    The function reads the original dataframe, extracts the impression_time column, converts it to datetime
    and writes the modified dataframe.
    Args:
        original_df_path: path to the original dataframe
        modified_df_path: path to the modified dataframe
    '''
    
    original = pl.read_parquet(original_df_path).select('impression_id','user_id','article_ids_inview','impression_time').explode('article_ids_inview') \
        .rename({'article_ids_inview':'article'})
    modified = pl.read_parquet(modified_df_path)

    new = modified.join(original, on=['impression_id','user_id', 'article'], how = 'left')
    
    assert new.select('impression_time_right', 'impression_time').with_columns(
             pl.col('impression_time_right').dt.date() - pl.duration(days=1)) \
                 .filter(pl.col('impression_time_right') != pl.col('impression_time')).shape[0] == 0
    
    new = new.drop('impression_time').rename({'impression_time_right': 'impression_time'})
  
    assert modified.drop('impression_time').equals(new.drop('impression_time'))
    
    new.write_parquet(modified_df_path)


def stack_slices(parquet_files: list[Path], save_path: Path, save_name: str, delete_all_slices:bool=False):
    assert len(parquet_files) > 0, 'No parquet files found in the directory'
    df = pl.read_parquet(parquet_files[0])
    for file in parquet_files[1:]:
        df = df.vstack(pl.read_parquet(file))
        
    print('Savig the final stacked dataframe...')
    df.write_parquet(save_path / f'{save_name}.parquet')
    if delete_all_slices:
        print(f'Deleting all {len(parquet_files)} slices...')
        for file in parquet_files:
            file.unlink()
        
def check_for_inf(df: pl.DataFrame):
    rows_with_inf = df.select(cs.numeric().is_infinite()).select(
        pl.sum_horizontal(pl.all()).alias('sum_infinite')
    ).filter(pl.col('sum_infinite') > 0).shape[0]

    cols_with_inf = df.select(cs.numeric().is_infinite())\
        .sum().transpose(include_header=True, header_name='column', column_names=['sum_infinite'])\
        .filter(pl.col('sum_infinite') > 0).to_dicts()
    return rows_with_inf, cols_with_inf