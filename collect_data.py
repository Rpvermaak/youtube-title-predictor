# import necessary libraries
from json import load
import os
from dotenv import load_dotenv
import pandas as pd
from requests import get

# load environment variables from .env file
load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")

CHANNELS_URL = "https://www.googleapis.com/youtube/v3/channels"
PLAYLIST_URL = "https://www.googleapis.com/youtube/v3/playlistItems"
VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"

def get_video_ids(playlist_id):

    params = {
        "part": "contentDetails",
        "playlistId": playlist_id,
        "maxResults": 50,
        "key": API_KEY,
    }

    response = get(PLAYLIST_URL, params=params, timeout=10)
    response.raise_for_status()

    page = response.json()
    video_ids = []
    while True:
        for item in page.get("items", []):
            video_ids.append(item["contentDetails"]["videoId"])
        if "nextPageToken" not in page:
            break
        params["pageToken"] = page["nextPageToken"]
        response = get(PLAYLIST_URL, params=params, timeout=10)
        response.raise_for_status()
        page = response.json()

    return video_ids

def get_video_details(video_ids):
    video_rows = []

    for start in range(0, len(video_ids), 50):
        chunk = video_ids[start:start + 50]
        params = {
            "part": "snippet,contentDetails,statistics,liveStreamingDetails",
            "id": ",".join(chunk),
            "key": API_KEY,
        }

        response = get(VIDEOS_URL, params=params, timeout=10)
        response.raise_for_status()
        page = response.json()
        for v in page["items"]:
            video_rows.append({
                "video_id": v["id"],
                "channel_id": v["snippet"]["channelId"],
                "title": v["snippet"]["title"],
                "published_at": v["snippet"]["publishedAt"],
                "duration": v["contentDetails"]["duration"],
                "view_count": int(v["statistics"].get("viewCount", 0)),
                "tags": "|".join(v["snippet"].get("tags", [])),
                "description": v["snippet"]["description"],
                "was_live": "liveStreamingDetails" in v,
            })

    return video_rows

def main():
    # load channels.csv
    channels = pd.read_csv("data/channels.csv")
    all_rows = []
    for i, row in channels.iterrows():
        print(f"{i+1}/{len(channels)} {row['title']}")
        try:
            ids = get_video_ids(row["uploads_playlist_id"])
            rows = get_video_details(ids)
            all_rows.extend(rows)
        except Exception as e:
            print(f"Error processing channel {row['title']}: {e}")
    videos_df = pd.DataFrame(all_rows)
    videos_df.to_csv("data/videos.csv", index=False)
    print(videos_df.shape)
    print(videos_df["channel_id"].value_counts().describe())

if __name__ == "__main__":
    main()