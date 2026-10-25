# import necessary libraries
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

def main():
    ids = get_video_ids("UUYAIbGWcLdohnpRh-YLXzRw")
    print(len(ids))

if __name__ == "__main__":
    main()