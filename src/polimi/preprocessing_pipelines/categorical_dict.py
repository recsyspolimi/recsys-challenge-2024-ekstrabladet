CATEGORICAL_COLUMNS = {
    '68f': ['device_type', 'is_sso_user', 'gender', 'is_subscriber', 'weekday',
                           'premium', 'category', 'sentiment_label', 'is_new_article', 'is_already_seen_article',
                           'MostFrequentCategory', 'MostFrequentWeekday', 'IsFavouriteCategory'],
    '94f': ['device_type', 'is_sso_user', 'gender', 'is_subscriber', 'weekday',
                           'premium', 'category', 'sentiment_label', 'is_new_article', 'is_already_seen_article',
                           'MostFrequentCategory', 'MostFrequentWeekday', 'IsFavouriteCategory'],
    '115f': ['device_type', 'is_sso_user', 'gender', 'is_subscriber', 'weekday',
                           'premium', 'category', 'sentiment_label', 'is_new_article', 'is_already_seen_article',
                           'MostFrequentCategory', 'MostFrequentWeekday', 'IsFavouriteCategory'],
    '127f': ['device_type', 'is_sso_user', 'gender', 'is_subscriber', 'weekday',
                           'premium', 'category', 'sentiment_label', 'is_new_article', 'is_already_seen_article',
                           'MostFrequentCategory', 'MostFrequentWeekday', 'IsFavouriteCategory'],
    '142f': ['device_type', 'is_sso_user', 'gender', 'is_subscriber', 'weekday',
                           'premium', 'category', 'sentiment_label', 'is_new_article', 'is_already_seen_article',
                           'MostFrequentCategory', 'MostFrequentWeekday', 'IsFavouriteCategory'],
    '147f': ['device_type', 'is_sso_user', 'gender', 'is_subscriber', 'weekday',
                           'premium', 'category', 'sentiment_label', 'is_new_article', 'is_already_seen_article',
                           'MostFrequentCategory', 'MostFrequentWeekday', 'IsFavouriteCategory'],
    # last version has also postcode and article_type
    'new': ['device_type', 'is_sso_user', 'gender', 'is_subscriber', 'weekday',
            'premium', 'category', 'sentiment_label', 'is_new_article', 'is_already_seen_article',
            'MostFrequentCategory', 'MostFrequentWeekday', 'IsFavouriteCategory', 'article_type', 'postcode'],
    'new_click': ['device_type', 'is_sso_user', 'gender', 'is_subscriber', 'weekday',
            'premium', 'category', 'sentiment_label', 'is_new_article', 'is_already_seen_article',
            'MostFrequentCategory', 'MostFrequentWeekday', 'IsFavouriteCategory', 'article_type', 'postcode'],
    'latest': ['device_type', 'is_sso_user', 'gender', 'is_subscriber', 'weekday',
                           'premium', 'category', 'sentiment_label', 'is_new_article', 'is_already_seen_article',
                           'MostFrequentCategory', 'MostFrequentWeekday', 'IsFavouriteCategory']
}

def get_categorical_columns(preprocessing_version):
    return CATEGORICAL_COLUMNS[preprocessing_version]