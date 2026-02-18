# python script to calculate similarity
import argparse
import gc
import os
import pickle
import tempfile
from pathlib import Path

import soundfile as sf
import torch
import wespeaker
import numpy as np

torch.set_num_threads(1)

def patch_torchaudio_for_soundfile():
    import torchaudio
    def custom_load(filepath, **kwargs):
        data, sr = sf.read(filepath)
        t = torch.tensor(data).float()
        if t.ndim == 1:
            t = t.unsqueeze(0)
        else:
            t = t.t()
        return t, sr
    torchaudio.load = custom_load


def _trim_to_temp(path: str, max_duration: float):
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


# calculate similarity (closer to 1 is more similar)
def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8))

# returns the closest match
def recognize(model, profiles: dict, audio_path: str, max_duration: float = None) -> tuple:
    use_path, tmp_path = _trim_to_temp(audio_path, max_duration)
    try:
        emb = model.extract_embedding(use_path)
        emb = np.asarray(emb).flatten()
    finally:
        if tmp_path:
            os.unlink(tmp_path)
    scores = {name: cosine_similarity(emb, prof) for name, prof in profiles.items()}
    best = max(scores.items(), key=lambda x: x[1])
    return best[0], best[1], scores

# run calculations & evaluate
def run_test(profiles_path: str = "speaker_profiles.pkl", language: str = "english", max_duration: float = None):
    patch_torchaudio_for_soundfile()
    model = wespeaker.load_model(language)
    print(f"Loading profiles from {profiles_path}...")
    with open(profiles_path, 'rb') as f:
        data = pickle.load(f)
    profiles = data['profiles']
    test_data = data.get('test_data', {})

    if not test_data or all(len(v) == 0 for v in test_data.values()):
        print("No test data in profiles. Run with --data_dir to evaluate on new data.")
        return
    print("\nEvaluating on test data...")
    correct = 0
    total = 0
    for true_speaker, paths in test_data.items():
        for path in paths:
            pred, conf, scores = recognize(model, profiles, path, max_duration=max_duration)
            total += 1
            if pred == true_speaker:
                correct += 1
            status = "✓" if pred == true_speaker else "✗"
            print(f"  {status} {Path(path).name}: true={true_speaker} pred={pred} conf={conf:.4%}")
            gc.collect()
    accuracy = correct / total if total else 0
    print(f"\n--- Results ---")
    print(f"Accuracy: {correct}/{total} = {accuracy:.2%}")
    gc.collect()


def run_test_on_directory(profiles_path: str, data_dir: str, language: str = "english", max_duration: float = None):
    from train_speaker_model import load_labeled_data, train_test_split

    print("Loading wespeaker model...")
    patch_torchaudio_for_soundfile()
    model = wespeaker.load_model(language)

    with open(profiles_path, 'rb') as f:
        data = pickle.load(f)
    profiles = data['profiles']

    speakers = load_labeled_data(data_dir)
    _, test_data = train_test_split(speakers, test_ratio=0.2)

    correct = 0
    total = 0
    for true_speaker, paths in test_data.items():
        if true_speaker not in profiles:
            continue
        for path in paths:
            pred, conf, _ = recognize(model, profiles, path, max_duration=max_duration)
            total += 1
            if pred == true_speaker:
                correct += 1
            print(f"  {'✓' if pred == true_speaker else '✗'} {path}: pred={pred} conf={conf:.4%}")
            gc.collect()

    if total:
        print(f"\nAccuracy: {correct}/{total} = {correct/total:.2%}")
    gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test speaker recognition model")
    parser.add_argument("--profiles", default="speaker_profiles.pkl", help="Path to speaker_profiles.pkl")
    parser.add_argument("--data_dir", default=None, help="Optional: test on data/speakers (uses train/test split)")
    parser.add_argument("--language", default="english", choices=["english", "chinese"])
    parser.add_argument("--max_duration", type=float, default=30, help="Use first N sec of each file (default 30, 0=full)")
    args = parser.parse_args()

    md = None if args.max_duration == 0 else args.max_duration
    if args.data_dir:
        run_test_on_directory(args.profiles, args.data_dir, args.language, max_duration=md)
    else:
        run_test(args.profiles, args.language, max_duration=md)
