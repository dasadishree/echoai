# python script to compare& output scores
import argparse
import gc
import os
import sys
import tempfile
import soundfile as sf
import torch
import torchaudio
import wespeaker
import numpy as np
from contextlib import redirect_stdout
import pickle
torch.set_num_threads(1)


def patch_torchaudio_for_soundfile():
    def custom_load(filepath, **kwargs):
        data, sample_rate = sf.read(filepath)
        tensor = torch.tensor(data).float()
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        else:
            tensor = tensor.t()
        return tensor, sample_rate
    torchaudio.load = custom_load


def cosine_similarity(emb1, emb2):
    return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8))


def _trim_and_get_path(path: str, max_duration: float):
    if max_duration is None or max_duration <= 0:
        return path, None
    with sf.SoundFile(path) as f:
        sr, n_total = f.samplerate, len(f)
        n_read = min(int(max_duration * sr), n_total)
        data = f.read(n_read)
    fd, tmp = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    sf.write(tmp, data, sr)
    del data
    return tmp, tmp


def recognize(profiles_path: str, audio_path: str, language: str = "english", max_duration: float = None):
    patch_torchaudio_for_soundfile()
    with open(os.devnull, 'w') as f, redirect_stdout(f):
        model = wespeaker.load_model(language)

    with open(profiles_path, 'rb') as f:
        data = pickle.load(f)
    profiles = data['profiles']
    del data  

    use_path, tmp_path = _trim_and_get_path(audio_path, max_duration)
    try:
        emb = model.extract_embedding(use_path)
        emb = np.asarray(emb).flatten()
    finally:
        if tmp_path:
            os.unlink(tmp_path)
    gc.collect()

    scores = {name: cosine_similarity(emb, prof) for name, prof in profiles.items()}
    best_name, best_score = max(scores.items(), key=lambda x: x[1])
    return best_name, best_score, scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recognize speaker from audio file")
    parser.add_argument("audio_file", help="Path to WAV file")
    parser.add_argument("--profiles", default="speaker_profiles.pkl", help="Path to speaker profiles")
    parser.add_argument("--language", default="english", choices=["english", "chinese"])
    parser.add_argument("--info", action="store_true", help="Show confidence and all scores")
    parser.add_argument("--max_duration", type=float, default=30, help="Use first N sec of audio (default 30, 0=full)")
    args = parser.parse_args()

    if not args.info:
        sys.stdout = open(os.devnull, 'w')

    md = None if args.max_duration == 0 else args.max_duration
    name, conf, all_scores = recognize(args.profiles, args.audio_file, args.language, max_duration=md)
    
    if not args.info:
        sys.stdout = sys.__stdout__

    print(f"\nHello {name.replace('_', ' ')}!")

    if args.info:
        print(f"Confidence: {conf:.4%}")
        print(f"{"SPEAKER NAME":<25} | {"SIMILARITY SCORE":>12}")
        print("-"*40)

        sorted_scores = dict(sorted(all_scores.items(), key=lambda item: item[1], reverse=True))
        for s_name, s_score in sorted_scores.items():
            print(f" {s_name:<25}: {s_score:.4%}")