import pandas as pd

numerical = ['duration_min', 'publish_hour', 'n_tags', 'description_length',
             'title_length', 'title_words', 'has_number', 'has_question',
             'has_exclamation', 'n_caps_words', 'has_emoji', 'has_separator']
categorical = ['publish_weekday']


def add_title_features(df):
    # Title statistics: used by train.py and predict.py
    df = df.copy()
    df['title_length'] = df['title'].str.len()
    df['title_words'] = df['title'].str.split().str.len()
    df['has_number'] = df['title'].str.contains(r'\d').astype(int)
    df['has_question'] = df['title'].str.contains('?', regex=False).astype(int)
    df['has_exclamation'] = df['title'].str.contains('!', regex=False).astype(int)
    df['n_caps_words'] = df['title'].apply(lambda t: sum(w.isupper() and len(w) > 1 for w in t.split()))
    df['has_emoji'] = df['title'].apply(lambda t: any(ord(c) > 0x2500 for c in t)).astype(int)
    df['has_separator'] = df['title'].str.contains(r'[|\-–]').astype(int)
    return df


def add_video_features(df):
    # Video details from the raw YouTube columns (only needed for training data)
    df = df.copy()
    df['published_at'] = pd.to_datetime(df['published_at'])
    df['duration_min'] = df['duration_s'] / 60
    df['publish_hour'] = df['published_at'].dt.hour          # UTC hour
    df['publish_weekday'] = df['published_at'].dt.day_name()
    df['n_tags'] = df['tags'].apply(lambda t: len(t.split('|')) if t else 0)
    df['description_length'] = df['description'].str.len()
    return df
