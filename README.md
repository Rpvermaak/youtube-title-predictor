# YouTube Travel Video Title Predictor

Pick the better title before you publish. Given 2–5 candidate titles for a travel video, this
service scores each one and ranks them by how likely the video is to **beat its channel's
typical views**.

Midterm project for [ML Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp).

## Problem description

We run [KarmicSteps](https://www.youtube.com/@KarmicSteps), a small couple travel channel
(~340 subscribers). Before each upload we have a few title ideas and no good way to choose between them.

**Question:** from information known *at upload time*, can a model tell which title gives a
video a better chance of outperforming the channel's usual performance?

**Why "beat the channel baseline" and not raw views:** a video with 5,000 views is a hit for a
small channel and a flop for a large one. So each video is compared with its own channel:

- **baseline** = the median views of the channel's previous 20 long-form videos
- **target** `beat_baseline` = 1 if the video's views are above its baseline, else 0

This removes channel size from the problem and gives a balanced target (~50/50).

**How the model is used:** the candidate titles share every other detail (duration, upload
day and hour, tags, description). The model scores each title and returns them ranked. The
ranking matters, not the exact probability.

## Data

- **Source:** YouTube Data API v3, collected on **2026-10-26**.
- **Channels:** 71 small English-language travel channels (1k–100k subscribers), found from
  seed channels and YouTube searches such as "Southeast Asia backpacking couple" and
  "Thailand travel vlog" (`explore_api.ipynb`). Data from 70 channels was collected; one failed.
- **Videos:** 21,814 videos (`data/videos.csv`), reduced by cleaning to **2,773 videos from
  29 channels** (`data/videos_clean.csv`):
  - long-form only (> 3 minutes), no live streams
  - at least 20 earlier videos to compute the baseline
  - aged 90 days to 3 years, so views have mostly settled
  - English titles only; channels with ≥ 20 remaining videos

The data is committed to the repo, so **no API key is needed** to reproduce the results.

| File | Contents |
|---|---|
| `data/channels.csv` | channel ID, name, subscribers, video count, how it was found |
| `data/videos.csv` | raw videos: title, publish time, duration, views, tags, description, live flag |
| `data/videos_clean.csv` | cleaned videos with `baseline` and `beat_baseline` (output of the notebook) |

## Approach

All features are known before upload. Post-upload numbers (likes, comments) are not used.

- **Title text:** TF-IDF on title words and word pairs (~1,070 terms)
- **Title statistics:** length, word count, contains a number / `?` / `!` / emoji / separator,
  number of ALL-CAPS words
- **Video details:** duration, publish weekday and hour (UTC), number of tags, description length

**Validation is grouped by channel:** 6 channels are held out as a test set, and
cross-validation (`GroupKFold`) never has the same channel in train and validation. This
checks that the model works on a *new* channel, like ours, and doesn't just learn channel habits.

## Results

Cross-validation AUC on unseen channels (5 folds, grouped by channel):

| Model | CV AUC |
|---|---|
| Logistic regression, no title text | 0.50 |
| **Logistic regression + title TF-IDF (C = 1)** | **0.546** |
| Random forest (max_depth=5) | 0.529 |
| XGBoost (eta=0.3, max_depth=3) | 0.540 |

The final model is **logistic regression with title TF-IDF**: it scores highest, it is the
simplest, and its coefficients are easy to read. On the 6 held-out test channels it reaches an
**AUC of 0.516**. It ranks better than chance for 4 of the 6 channels (0.53–0.58).

**What the model learned:**
- Without the title text, the model is no better than chance. Duration, upload time, tags
  and description length don't predict whether a video beats its own baseline.
- Practical, informational words help: *itinerary*, *cost*, *flight*, *worth*, *shopping*, *new*.
  Titles with a question mark beat the baseline 57% of the time vs 49% without.
- Generic vlog labels hurt: *vlog*, *episode*, *youtube*.

**Honest conclusion:** title wording carries a small but real signal. Most of a video's
performance depends on things this model can't see (thumbnail, topic, timing, the algorithm).
Use it as a second opinion when choosing between titles, not as a view forecast.

## How to run

### Requirements

- [uv](https://docs.astral.sh/uv/getting-started/installation/) (manages Python and the packages)
- [Docker](https://docs.docker.com/get-docker/) (only for the Docker option)

### 1. Get the code and install dependencies

```bash
git clone https://github.com/Rpvermaak/youtube-title-predictor.git
cd youtube-title-predictor
uv sync
```

`uv sync` installs the exact package versions from `uv.lock`, including the notebook packages
(Jupyter, matplotlib, XGBoost) in the `dev` group.

### 2. Train the model (optional)

The trained model `model.bin` is already in the repo. To retrain it from `data/videos_clean.csv`:

```bash
uv run python train.py
```

Expected output:

```
Test AUC: 0.516
Model saved to model.bin
```

### 3. Run the service locally

```bash
uv run python predict.py
```

The service runs on `http://localhost:9696`. Leave it running and test it from a second terminal
(see [Example request](#example-request)).

### 4. Run the service with Docker

```bash
docker build -t youtube-title-predictor .
docker run -it --rm -p 9696:9696 youtube-title-predictor
```

The image installs only the service packages (FastAPI, pandas, scikit-learn, uvicorn) with the
locked versions from `uv.lock`.

### Example request

```bash
curl -X POST localhost:9696/predict \
  -H "Content-Type: application/json" \
  -d '{
    "titles": ["Vlog | Episode 12", "What a week in Hanoi really costs", "Is Da Nang worth it?"],
    "duration_min": 18,
    "publish_weekday": "Saturday",
    "publish_hour": 17,
    "n_tags": 12,
    "description_length": 600
  }'
```

Response, best title first:

```json
{
  "ranking": [
    {"title": "Is Da Nang worth it?", "probability": 0.751},
    {"title": "What a week in Hanoi really costs", "probability": 0.513},
    {"title": "Vlog | Episode 12", "probability": 0.291}
  ]
}
```

| Field | Type | Notes |
|---|---|---|
| `titles` | list of strings | 1–5 candidate titles |
| `duration_min` | number | video length in minutes |
| `publish_weekday` | string | `Monday` … `Sunday` |
| `publish_hour` | integer | 0–23, UTC |
| `n_tags` | integer | optional, default 0 |
| `description_length` | integer | optional, characters, default 0 |

You can also try the API in the browser at **http://localhost:9696/docs**.
`GET /health` returns `{"status": "ok"}`.

### Notebook

```bash
uv run jupyter notebook notebook.ipynb
```

Covers data cleaning, the target, EDA, feature engineering, model selection and the final test.

### Collecting the data again (optional)

Needs a YouTube Data API key. Copy `.env.example` to `.env`, add your key, then run:

```bash
uv run python collect_data.py
```

New data will give somewhat different results, because view counts keep changing.

## Project structure

```
├── notebook.ipynb        # cleaning, EDA, feature engineering, model selection, final test
├── explore_api.ipynb     # finding the channels (scratch notebook)
├── collect_data.py       # downloads the videos of all channels via the YouTube API
├── features.py           # feature code shared by train.py and predict.py
├── train.py              # trains the final model and saves model.bin
├── predict.py            # FastAPI service
├── model.bin             # trained model (DictVectorizer, TF-IDF, logistic regression)
├── Dockerfile
├── pyproject.toml, uv.lock
└── data/                # channels, raw videos, cleaned videos
```

## Limitations and future work

- **Small dataset:** 2,773 videos from 29 channels. Small effects such as title wording need
  many more channels to show clearly. Collecting 300+ channels would help most.
- **Channel selection bias:** channels were found through YouTube search, which favours
  channels that are currently growing. This is why videos younger than 90 days were removed.
- **Channel-specific words:** TF-IDF learns exact words, and some (place names, a channel's
  series name) reflect one channel's habits. Pretrained sentence embeddings would capture
  meaning, like "practical cost information", which may carry over better to new channels.
- **Yes/no target:** beating the baseline by 1% counts the same as beating it 10×. Predicting
  log(views ÷ baseline) as regression would keep that information.
- **No thumbnail:** the thumbnail probably matters as much as the title, but it isn't used.
- **Publish hour is UTC**, so it means different local times for different channels.
