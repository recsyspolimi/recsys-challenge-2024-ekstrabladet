import polars as pl
from typing_extensions import List, Union


def get_norm_expression(norm_columns: List[str], over: Union[str, pl.Expr] = 'impression_id',
                         norm_type: str = 'infinity', suffix_name: str = '_impression'):
    if norm_type == 'infinity':
        return [(pl.col(c) / pl.col(c).max().over(over)).alias(f'{c}_l_inf{suffix_name}') for c in norm_columns]
    elif norm_type == 'l1':
        return [(pl.col(c) / pl.col(c).sum().over(over)).alias(f'{c}_l1{suffix_name}') for c in norm_columns]
    elif norm_type == 'l2':
        return [(pl.col(c) / pl.col(c).pow(2).sum().over(over)).alias(f'{c}_l2{suffix_name}') for c in norm_columns]
    else:
        raise NotImplementedError('Selected norm type is not implemented, choose between [l1, l2, infinity]')
    

def get_diff_norm_expression(norm_columns: List[str], over: Union[str, pl.Expr] = 'impression_id', 
                              diff_type: str = 'median', quantile_p: float = 0.8, suffix_name: str = '_impression'):
    if diff_type == 'median':
        return [(pl.col(c) -  pl.col(c).median().over(over)).alias(f'{c}_minus_median{suffix_name}') for c in norm_columns]
    elif diff_type == 'mean':
        return [(pl.col(c) / pl.col(c).mean().over(over)).alias(f'{c}_minus_mean{suffix_name}') for c in norm_columns]
    elif diff_type == 'quantile':
        if quantile_p > 1 or quantile_p < 0:
            raise ValueError('quantile_p must be between 0 and 1')
        return [(pl.col(c) / pl.col(c).quantile(quantile_p).over(over)).alias(f'{c}_minus_quantile{int(100*quantile_p)}{suffix_name}') 
                for c in norm_columns]
    else:
        raise NotImplementedError('Selected diff type is not implemented, choose between [median, mean, quantile]')
    
    
def get_list_diversity_expression(columns: List[str], over: Union[str, pl.Expr] = 'impression_id', suffix_name: str = '_impression'):
    return [pl.col(c).n_unique().over(over).alias(f'{c}_diversity{suffix_name}') for c in columns]


def get_list_rank_expression(columns: List[str], over: Union[str, pl.Expr] = 'impression_id', 
                              suffix_name: str = '_impression', descending: bool = False):
    return [pl.col(c).rank(method='min', descending=descending).over(over).alias(f'{c}_rank{suffix_name}')
            for c in columns]


def get_group_stats_expression(stats_columns: List[str], over: Union[str, pl.Expr] = 'impression_id', stat_type: str = 'std', 
                                quantile_p: float = 0.8, suffix_name: str = '_impression'):
    if stat_type == 'median':
        return [pl.col(c).median().over(over).alias(f'median{suffix_name}_{c}') for c in stats_columns]
    elif stat_type == 'mean':
        return [pl.col(c).mean().over(over).alias(f'median{suffix_name}_{c}') for c in stats_columns]
    elif stat_type == 'std':
        return [pl.col(c).std().over(over).alias(f'std{suffix_name}_{c}') for c in stats_columns]
    elif stat_type == 'skew':
        return [pl.col(c).skew().over(over).alias(f'skew{suffix_name}_{c}') for c in stats_columns]
    elif stat_type == 'kurtosis':
        return [pl.col(c).kurtosis().over(over).alias(f'entropy{suffix_name}_{c}') for c in stats_columns]
    elif stat_type == 'entropy':
        return [pl.col(c).kurtosis().over(over).alias(f'kurtosis{suffix_name}_{c}') for c in stats_columns]
    elif stat_type == 'quantile':
        if quantile_p > 1 or quantile_p < 0:
            raise ValueError('quantile_p must be between 0 and 1')
        return [pl.col(c).quantile(quantile_p).over(over).alias(f'quantile{int(100*quantile_p)}{suffix_name}_{c}') for c in stats_columns]
    else:
        raise NotImplementedError(
            'Selected statistic is not implemented, choose between [median, mean, quantile, std, skew, kurtosis, entropy]'
        )