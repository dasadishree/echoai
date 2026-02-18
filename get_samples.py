# get samples for trainijng data
import os
import yt_dlp
from yt_dlp.utils import download_range_func

# known url clips from youtube
training_clips = [
    {"speaker": "sample_002.wav", "url": "", "start":0, "end": 0} #angelina jolie
]

os.makedirs("labeled_samples", exist_ok=True)

for clip in training_clips:
    output_filename = f"labeled_samples/{clip['speaker']}.%(ext)s"
    ydl_opts={
        'format': 'bestaudio/best',
        'download_ranges': download_range_func(None, [(clip['start'], clip['end'])]),        
        'force_keyframes_at_cuts': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
        }],
        'outtmpl': output_filename,
        'quiet': False, 
    }
    print(f"\n--- Extracting {clip['speaker']} ({clip['start']}) ---")

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([clip['url']])
    except Exception as e:
            print(f"Failed to download {clip['speaker']}: {e}")

print("All clips processed")