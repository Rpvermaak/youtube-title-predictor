import pickle
from typing import Literal

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field

from features import numerical, categorical, add_title_features


# Request: 1–5 candidate titles plus the details they share
class Video(BaseModel):
    titles: list[str] = Field(..., min_length=1, max_length=5)
    duration_min: float = Field(..., gt=0)
    publish_weekday: Literal['Monday', 'Tuesday', 'Wednesday', 'Thursday',
                             'Friday', 'Saturday', 'Sunday']
    publish_hour: int = Field(..., ge=0, le=23)     # UTC hour
    n_tags: int = Field(0, ge=0)
    description_length: int = Field(0, ge=0)


# Response: one score per title, best first
class TitleScore(BaseModel):
    title: str
    probability: float


class PredictResponse(BaseModel):
    ranking: list[TitleScore]


app = FastAPI(title='youtube-title-predictor')

with open('model.bin', 'rb') as f_in:
    dv, tfidf, model = pickle.load(f_in)


def predict_titles(video):
    # One row per candidate title, with the shared video details copied to every row
    df = pd.DataFrame({'title': video.titles})
    df['duration_min'] = video.duration_min
    df['publish_weekday'] = video.publish_weekday
    df['publish_hour'] = video.publish_hour
    df['n_tags'] = video.n_tags
    df['description_length'] = video.description_length

    df = add_title_features(df)

    X = dv.transform(df[categorical + numerical].to_dict(orient='records'))
    X_text = tfidf.transform(df.title).toarray()
    X = np.hstack([X, X_text])

    return model.predict_proba(X)[:, 1]


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.post('/predict')
def predict(video: Video) -> PredictResponse:
    probabilities = predict_titles(video)

    ranking = [TitleScore(title=t, probability=round(float(p), 3))
               for t, p in zip(video.titles, probabilities)]
    ranking.sort(key=lambda r: r.probability, reverse=True)

    return PredictResponse(ranking=ranking)


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=9696)
