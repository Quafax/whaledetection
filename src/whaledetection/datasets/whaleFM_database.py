import os
import pandas as pd
import requests
import io
from pydub import AudioSegment

assets_url = "https://raw.githubusercontent.com/zooniverse/WhaleFM/master/csv/whale_fm_anon_04-03-2015_assets.csv"


def sanitize_name(name: str) -> str:
    name = str(name).strip()
    bad_chars = r'<>:"/\|?*'
    for ch in bad_chars:
        name = name.replace(ch, "_")
    return name

def export_whalefm_database(base_out_dir: str):

    os.makedirs(base_out_dir, exist_ok=True)

    print("Loading assets.csv ...")
    df = pd.read_csv(assets_url)

    session = requests.Session()

    total = len(df)
    downloaded = 0

    for i, row in df.iterrows():

        audio_url = str(row["location"]).strip()

        if audio_url == "" or audio_url == "NULL":
            continue

        whale_type = sanitize_name(row["whale_type"])

        out_dir = os.path.join(base_out_dir, whale_type)
        os.makedirs(out_dir, exist_ok=True)

        filename = f"{i:06d}.wav"
        out_path = os.path.join(out_dir, filename)

        try:
            r = session.get(audio_url, timeout=60)
            r.raise_for_status()

            audio = AudioSegment.from_file(io.BytesIO(r.content), format="mp3")

            audio.export(out_path, format="wav")

            downloaded += 1

            if downloaded % 100 == 0:
                print(f"{downloaded} files saved")

        except Exception as e:
            print(f"Error at {audio_url}")
            print(e)
    print(f"{downloaded} files saved.")