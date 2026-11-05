# YouTube Travel Video Title Predictor — Midterm Project Design

_Created 2026-10-03, revised 2026-10-17._

## Context

ML Zoomcamp (DataTalksClub) midterm project. The project is peer-reviewed
against a fixed 16-point rubric (see `projects/README.md` in the course repo),
and separately requires reviewing 3 peers' projects or the submission fails
regardless of score.

This project is also meant to be genuinely useful for **KarmicSteps**, a
couples travel YouTube channel the user co-runs (~340 subscribers at time of
writing), and to double as a portfolio piece demonstrating creator-analytics
modeling skills. Viewers in the channel's chat regularly ask for / suggest
title ideas; this tool is how the user will pick between them.

It is also a **learning project**: stay close to the techniques taught in the
course (modules 1–6) rather than reaching for heavier tooling.

**Who builds it:** the user writes all the code themselves, with Claude
guiding (explaining, reviewing, unblocking). The implementation plan should
be a sequence of learnable tasks for the user, notebook-first, not code for
an agent to paste in.

**Deadline:** project submission due **2026-11-09** (Mon, 23:00 UTC); peer reviews of 3 projects due **2026-11-16**. (Corrected 2026-10-26 — 16 Nov is the peer-review deadline, not submission.)

It will live in its **own standalone GitHub repo** (not a subfolder of this
course repo), per the course FAQ's recommendation for a real project title
and better portfolio visibility.

## Goal and primary use

Predict, from information known **at upload**, whether a long-form travel
video will outperform its own channel's recent typical performance.

**Primary use case:** before publishing a video, the user types in **2–3
candidate titles** plus the video's shared details (duration, planned upload
day/hour etc.). The service scores each title and
returns them **ranked**, highest probability of beating the channel's typical
performance first. The relative ordering of titles matters more than the
absolute probability.

Consequence for design: since every non-title input is identical across the
candidates, **the title must carry real signal in the model** — not only
surface stats (length, caps) but the words themselves.

## Non-goals / anti-goals

- Not a precise view-count forecast.
- Not a title *generator* — it scores titles the user provides.
- Failure even if it "works": if the ranking is driven only by title length
  or punctuation (i.e. it ignores what the title says), it doesn't serve the
  use case. Check this in the notebook (see Evaluation).

## Data

- **Source**: YouTube Data API v3. API key stored in `.env`, never committed
  (`.gitignore` must cover it); `.env.example` documents the variable.
- **Channel scope**: 50–100 **small travel channels, ~1k–100k subscribers**
  (lowered from earlier "small/mid-sized" because KarmicSteps is at ~340 and
  the training data should be as close to that as practical). Prefer
  vlog/couples/budget-travel style channels similar to KarmicSteps. Seed the
  list with the **channels the user already has in mind** (a handful), then
  top up to 50–100 with `search.list` channel searches (queries like "couple
  travel vlog", "budget travel vlog"; ~20 searches ≈ 2,000 units) plus
  YouTube's related/featured channels of the seeds, then filter by
  subscriber count and English language. Seeds should be reviewed by the
  user before collection. Commit the
  final list as `data/channels.csv` (channel ID, name, subscriber count,
  how it was found).
- **Collection method**: to stay within the default 10,000-unit daily quota,
  use `channels.list` (1 unit) → `playlistItems.list` on the uploads playlist
  (1 unit / 50 videos) → batched `videos.list` (1 unit / 50 videos). For 100
  channels × ~300 videos this is roughly 1,500 units. `search.list` (100
  units/call) is used only for channel discovery, sparingly.
- **Fields pulled**: video ID, channel ID, title, description, tags, duration,
  publish datetime, live-broadcast flag, view count; channel subscriber count.
- **Filters**:
  - **Long-form only**: drop videos with duration ≤ 180 s (Shorts can be up to
    3 minutes) and drop live streams / premieres-as-live.
  - Age **> 90 days** (raised from 30 days after EDA on 2026-10-31: videos
    aged 30–90 days beat their baseline 65% of the time vs ~50% for every older
    band, likely a selection effect of finding channels via search while they
    are growing) and **≤ ~3 years** (keeps
    very old videos from dominating and limits drift in what titles work).
- **Per-channel floor**: only include channels with at least ~20 qualifying
  videos after filtering.
- **English titles only**: prefer English-language
  channels when curating, and drop non-English titles during cleaning (e.g.
  `langdetect` or a simple non-ASCII-ratio heuristic). Mixed languages would
  fragment the TF-IDF vocabulary and make "top title words" misleading.
- **Reproducibility**: commit the collected dataset as `data/videos.csv` so
  peer reviewers don't need an API key. `collect_data.py` re-collects only if
  an API key is present; record the collection date in the README. Before
  committing data to a public repo, check the YouTube API Services terms on
  storing/redistributing API data.
- **Fallback** if collection stalls past week 1: the Kaggle "YouTube
  trending videos" dataset (weaker fit — big channels, not travel-specific).
- **KarmicSteps**: collected separately and **excluded from training/CV**.
  Used only as a qualitative real-world check (it likely has too few videos
  for a reliable metric).

## Target

Binary classification: `beat_baseline = 1` if the video's view count exceeds
the **median view count of the same channel's previous 20 qualifying
uploads** (rolling, prior videos only); else 0.

- Rationale: normalizes away channel size, and the rolling window handles
  channel growth (an all-time median would label old videos of a growing
  channel as failures). Uses only information that existed at upload time.
- A channel's first ~20 qualifying videos have no full baseline: drop them
  (or use a shorter window with a minimum of ~10 — builder's call, document
  it).
- Expect roughly balanced classes; confirm in EDA.

Alternative considered and rejected for v1: regression on
log(views / subscribers). The API only returns *current* subscriber count, not
the count at upload, so this target is distorted for growing channels and is
harder to act on. Mention in README as a future direction.

## Features

All known at upload time.

- **Title text**: TF-IDF on title words (unigrams + bigrams, lowercase,
  modest `min_df`) — the core of the title-ranking use case.
- **Title stats**: character length, word count, contains a number, contains
  `?`, contains `!`, count of ALL-CAPS words, contains emoji, contains `|` or
  `-` separator.
- **Video details**: duration (minutes, possibly log), publish day of week,
  publish hour (UTC, or bucketed), number of tags, description length.

Excluded: Short/long-form flag (all data is long-form now), thumbnails,
any post-upload engagement, subscriber count (the API only gives the current
count, and the target already compares each video with its own channel's baseline).

## Leakage rules

- No likes, comments, or any post-upload engagement metric as features.
- No channel-level statistics computed using the video being predicted or
  later videos (the rolling baseline already uses only prior videos).
- TF-IDF / scalers / encoders fitted inside the pipeline on training folds
  only (use a scikit-learn `Pipeline` / `ColumnTransformer`, or
  `DictVectorizer` fitted on train only).
- **Validation is grouped by channel** (`GroupKFold` on channel ID) so the
  model is always scored on channels it has never seen — which is exactly the
  situation when scoring KarmicSteps titles.

## Modeling

1. Baseline: logistic regression (title stats + metadata, no text)
2. Logistic regression with TF-IDF title text + other features
3. Random forest
4. XGBoost

Tune hyperparameters with channel-grouped CV. Compare on **ROC AUC**. Select
and justify the best model. (Random forest / XGBoost on sparse TF-IDF may need
reduced vocabulary or dimension reduction — builder's call; logistic
regression with TF-IDF is a strong, course-aligned candidate.)

## Evaluation

- Main metric: ROC AUC on channel-grouped CV and on a held-out set of
  channels.
- **Title-signal check**: show that the text features improve AUC over the
  no-text baseline (model 1 vs model 2), and list the most positive/negative
  title words from the logistic regression as feature importance.
- **Ranking sanity check**: within held-out channels, does the model's score
  order videos better than chance? (e.g. per-channel AUC averaged.)
- **KarmicSteps check**: score KarmicSteps' past videos and discuss in the
  README whether the ranking looks sensible. Qualitative only.

**Choosing the final model**: report both pooled
channel-grouped AUC and mean per-channel AUC. Pick the model with the best
pooled AUC, but if a model that uses title text is within ~0.01 AUC of the
top, prefer it — the use case only varies the title.

**If the title text adds little signal**: ship anyway.
This is a valid finding for a midterm; report the size of the gain honestly
in the README (e.g. "title words add +0.02 AUC over metadata alone") and
list it as a limitation. Don't redesign mid-project. Add a checkpoint at the
end of week 2 to look at the model 1 vs model 2 gap so it's known early.

## Service (predict.py, FastAPI)

- `GET /health`
- `POST /predict` — request: shared video details (duration, upload
  day/hour, tag count, description length) plus a list of
  1–5 candidate titles. Response: titles with their probability, sorted
  highest first.

Example request:

```json
{
  "titles": ["We tried €1 street food in Hanoi", "Hanoi on a budget | Vietnam vlog"],
  "duration_min": 18,
  "publish_weekday": "Saturday",
  "publish_hour": 17,
  "n_tags": 12,
  "description_length": 600
}
```

## Course deliverables (graded against the 16-point rubric)

- `README.md`: problem description with context (2 pts), dataset description
  and reproduction instructions, how to run locally and via Docker, example
  API request/response, known limitations.
- `notebook.ipynb`: data cleaning, extensive EDA (target distribution, feature
  distributions, missing values, feature importance — 2 pts; the rubric's
  text-specific bar also expects **frequent title words / word clouds**,
  ideally split by `beat_baseline` 0 vs 1), model selection and tuning
  process.
- `train.py`: trains the final model on the committed dataset and saves it
  (1 pt export + 1 pt reproducibility).
- `predict.py`: FastAPI service as above (1 pt deployment).
- Dependencies via `uv` (`pyproject.toml` + `uv.lock`) with install/run
  instructions (2 pts).
- `Dockerfile` that builds and runs the service, with instructions (2 pts).
- **Optional bonus**: cloud deployment with URL and/or screenshot/video.

## Suggested repo structure

```
collect_data.py     # YouTube API collection script
data/
  channels.csv
  videos.csv
notebook.ipynb
train.py
predict.py
model.bin
pyproject.toml
uv.lock
Dockerfile
README.md
.env.example
.gitignore          # excludes .env
```

## Plan (submission due 2026-11-09, peer reviews due 2026-11-16)

If time runs short, cut in this order: cloud deployment bonus → random
forest → hour/weekday bucketing refinements. Never cut: grouped CV, README,
Docker, fresh-clone test.

- **Week 1**: API key, channel list curation, `collect_data.py`, collect and
  commit data, compute the rolling-median target.
- **Week 2**: EDA and feature engineering (incl. TF-IDF).
- **Week 3**: Train and tune the 4 models with channel-grouped CV; export
  `train.py`.
- **Week 4**: FastAPI service, uv, Docker, optional cloud deploy, README,
  fresh-clone test, peer-review prep.

## Out of scope for v1

- Regression target (log views/subscriber) — future direction only.
- Thumbnail image analysis.
- Sentence embeddings / transformer text models (keep it course-aligned;
  possible future upgrade).
- Any post-upload engagement signals as features.
