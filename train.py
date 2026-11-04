import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from features import numerical, categorical, add_title_features, add_video_features

# Parameters (chosen in notebook.ipynb, section 5)
C = 1.0
output_file = 'model.bin'


# Data
df = pd.read_csv('data/videos_clean.csv')
df['tags'] = df['tags'].fillna('')
df['description'] = df['description'].fillna('')

df = add_video_features(df)
df = add_title_features(df)

# Same channel-grouped split as the notebook
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=1)
train_idx, test_idx = next(gss.split(df, groups=df.channel_id))
df_full_train = df.iloc[train_idx].reset_index(drop=True)
df_test = df.iloc[test_idx].reset_index(drop=True)


# Training
def train(df_train, y_train, C=1.0):
    dicts = df_train[categorical + numerical].to_dict(orient='records')
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=5, stop_words='english')
    X_text = tfidf.fit_transform(df_train.title).toarray()
    X_train = np.hstack([X_train, X_text])

    model = LogisticRegression(solver='liblinear', C=C, max_iter=1000)
    model.fit(X_train, y_train)
    return dv, tfidf, model


def predict(df, dv, tfidf, model):
    dicts = df[categorical + numerical].to_dict(orient='records')
    X = dv.transform(dicts)
    X_text = tfidf.transform(df.title).toarray()
    X = np.hstack([X, X_text])
    return model.predict_proba(X)[:, 1]


# Check on the test channels (should match the notebook: 0.516)
dv, tfidf, model = train(df_full_train, df_full_train.beat_baseline.values, C=C)
y_pred = predict(df_test, dv, tfidf, model)
print('Test AUC: %.3f' % roc_auc_score(df_test.beat_baseline, y_pred))

# Final model: trained on all channels
dv, tfidf, model = train(df, df.beat_baseline.values, C=C)

with open(output_file, 'wb') as f_out:
    pickle.dump((dv, tfidf, model), f_out)

print(f'Model saved to {output_file}')
