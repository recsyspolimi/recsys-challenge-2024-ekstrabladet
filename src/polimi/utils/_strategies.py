import polars as pl
import datetime


def _behaviors_to_history(behaviors: pl.DataFrame) -> pl.DataFrame:
        return behaviors.sort('impression_time').select('user_id', 'impression_time', 'next_scroll_percentage', 'article_ids_clicked', 'next_read_time')\
                .rename({'impression_time': 'impression_time_fixed', 
                        'article_ids_clicked': 'article_id_fixed', 
                        'next_read_time': 'read_time_fixed', 
                        'next_scroll_percentage': 'scroll_percentage_fixed'})\
                .explode('article_id_fixed').group_by('user_id').agg(pl.all())
        

def moving_window_split_iterator(history: pl.DataFrame, behaviors: pl.DataFrame, window:int=4, window_val:int=2, stride:int=2, verbose=True):
    assert behaviors['impression_time'].is_sorted()
    
    
    all_dates = history['impression_time_fixed'].explode().dt.date().unique().append(
        behaviors['impression_time'].dt.date().unique()
    ).unique().sort().to_list()
    all_dates_map = {date: i for i, date in enumerate(all_dates)}
    if verbose:
        print(f'Date range: [{all_dates[0]}:{all_dates_map[all_dates[0]]} - {all_dates[-1]}:{all_dates_map[all_dates[-1]]}]')
    
    history_window_train_start_date = history['impression_time_fixed'].explode().min().date()    
    start_window_train_behavior_date = behaviors['impression_time'].min().date()
    last_date = behaviors['impression_time'].max().date()
    i = 0
    while  start_window_train_behavior_date + datetime.timedelta(days=window + window_val) <= last_date:
        end_window_train_behavior_date = start_window_train_behavior_date + datetime.timedelta(days=window)
        start_window_val_behavior_date  = end_window_train_behavior_date
        end_window_val_behavior_date = start_window_val_behavior_date + datetime.timedelta(days=window_val)
        
        history_window_val_start_date = history_window_train_start_date + datetime.timedelta(days=7)


        if verbose:
            print(f'Fold {i}: ')
            print(f'Train: [{all_dates_map[history_window_train_start_date]} - {all_dates_map[start_window_train_behavior_date]} - {all_dates_map[end_window_train_behavior_date]}]')
            print(f'Validation: [{all_dates_map[history_window_val_start_date]} - {all_dates_map[start_window_val_behavior_date]} - {all_dates_map[end_window_val_behavior_date]}]')
        
            
        behaviors_k_train = behaviors.filter(
            pl.col('impression_time') >= datetime.datetime.combine(start_window_train_behavior_date, datetime.time(7, 0, 0)),
            pl.col('impression_time') < datetime.datetime.combine(end_window_train_behavior_date, datetime.time(7, 0, 0)),
        )
        
        history_k_train = history.explode(pl.all().exclude('user_id')).filter(
            pl.col('impression_time_fixed') >= datetime.datetime.combine(history_window_train_start_date, datetime.time(7, 0, 0)),
            pl.col('impression_time_fixed') < datetime.datetime.combine(
                history_window_train_start_date + datetime.timedelta(days=21), datetime.time(7, 0, 0)),
        ).group_by('user_id').agg(pl.all())

        behaviors_k_val = behaviors.filter(
            pl.col('impression_time') >= datetime.datetime.combine(start_window_val_behavior_date, datetime.time(7, 0, 0)),
            pl.col('impression_time') < datetime.datetime.combine(end_window_val_behavior_date, datetime.time(7, 0, 0)),
        )
        
        history_k_val = history.explode(pl.all().exclude('user_id')).filter(
            pl.col('impression_time_fixed') >= datetime.datetime.combine(history_window_val_start_date, datetime.time(7, 0, 0)),
            pl.col('impression_time_fixed') < datetime.datetime.combine(
                history_window_val_start_date + datetime.timedelta(days=21), datetime.time(7, 0, 0)),
        ).group_by('user_id').agg(pl.all())
        
        start_window_train_behavior_date += datetime.timedelta(days=stride)
        history_window_train_start_date += datetime.timedelta(days=stride)
        i+=1
        
        yield history_k_train, behaviors_k_train, history_k_val, behaviors_k_val