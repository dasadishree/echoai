#python script to train model based on speaker training audios
import gc
import os
import tempfile
import pickle
import argparse
from pathlib import Path

import soundfile as sf
import torch
import torchaudio
import wespeaker
import numpy as np

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

# loads the training data
def load_labeled_data(data_dir: str):
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    flat_wavs = list(data_path.glob('*.wav')) + list(data_path.glob('*.WAV'))
    if flat_wavs:
        speakers = {}
        for f in flat_wavs:
            name = f.stem 
            speakers.setdefault(name, []).append(str(f))
        return speakers

    speakers = {}
    for speaker_dir in sorted(data_path.iterdir()):
        if speaker_dir.is_dir() and not speaker_dir.name.startswith('.'):
            wav_files = list(speaker_dir.glob('*.wav')) + list(speaker_dir.glob('*.WAV'))
            if wav_files:
                speakers[speaker_dir.name] = [str(f) for f in sorted(wav_files)]

    return speakers

# split between training and testing
def train_test_split(speakers: dict, test_ratio: float = 0.2, min_train: int = 1, min_test: int = 1):
    train_data = {}
    test_data = {}

    for name, files in speakers.items():
        n = len(files)
        if n < min_train + min_test:
            if n >= 2:
                train_data[name] = files[:-1]
                test_data[name] = files[-1:]
            else:
                train_data[name] = files
                test_data[name] = []
        else:
            n_test = max(min_test, int(n * test_ratio))
            n_train = n - n_test
            train_data[name] = files[:n_train]
            test_data[name] = files[-n_test:]

    return train_data, test_data

# loads shortened audio 
def _load_trimmed_audio(path: str, max_duration_sec: float):
    with sf.SoundFile(path) as f:
        sr = f.samplerate
        n_total = len(f)
        n_needed = int(max_duration_sec * sr) if max_duration_sec else n_total
        n_read = min(n_needed, n_total)
        data = f.read(n_read)
    return data, sr

# extract data from the audioos
def extract_and_average_embeddings(model, file_paths: list, max_duration: float = None) -> np.ndarray:
    running_sum = None
    count = 0
    for path in file_paths:
        try:
            if max_duration is not None:
                data, sr = _load_trimmed_audio(path, max_duration)
                fd, tmp_path = tempfile.mkstemp(suffix=".wav")
                try:
                    os.close(fd)
                    sf.write(tmp_path, data, sr)
                    del data 
                    emb = model.extract_embedding(tmp_path)
                finally:
                    os.unlink(tmp_path)
            else:
                emb = model.extract_embedding(path)
            if emb is not None and len(emb) > 0:
                emb = np.asarray(emb).flatten()
                if running_sum is None:
                    running_sum = emb.copy()
                else:
                    running_sum += emb
                count += 1
        except Exception as e:
            print(f"  Warning: Could not process {path}: {e}")
        gc.collect()

    if count == 0:
        return None
    return running_sum / count

# trains the ai model
def train(data_dir: str = "data/speakers", output_path: str = "speaker_profiles.pkl",
          test_ratio: float = 0.2, language: str = "english", max_duration: float = None):
    print("Loading wespeaker model...")
    patch_torchaudio_for_soundfile()
    model = wespeaker.load_model(language)

    print(f"Loading labeled data from {data_dir}...")
    speakers = load_labeled_data(data_dir)
    if not speakers:
        raise ValueError(f"No speaker data found in {data_dir}. "
                        "Expected structure: data/speakers/{{name}}/*.wav")

    train_data, test_data = train_test_split(speakers, test_ratio=test_ratio)

    if max_duration:
        print(f"Using first {max_duration}s of each file (faster for long recordings)")
    print("\nBuilding voice profiles...")
    profiles = {}
    for name, paths in train_data.items():
        print(f"  {name}: {len(paths)} training samples")
        emb = extract_and_average_embeddings(model, paths, max_duration=max_duration)
        if emb is not None:
            profiles[name] = emb
        else:
            print(f"  Warning: No valid embeddings for {name}, skipping")
        gc.collect()

    if not profiles:
        raise ValueError("No valid speaker profiles could be built.")

    # save profiles
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, 'wb') as f:
        pickle.dump({
            'profiles': profiles,
            'speakers': list(profiles.keys()),
            'train_data': {k: v for k, v in train_data.items() if k in profiles},
            'test_data': {k: v for k, v in test_data.items() if k in profiles},
        }, f)

    print(f"\nSaved {len(profiles)} speaker profiles to {output_path}")
    return profiles, test_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train speaker recognition from labeled data")
    parser.add_argument("--data_dir", default="data/speakers", help="Directory with data/speakers/{name}/*.wav")
    parser.add_argument("--output", default="speaker_profiles.pkl", help="Output profiles file")
    parser.add_argument("--test_ratio", type=float, default=0.2, help="Fraction of data for testing")
    parser.add_argument("--language", default="english", choices=["english", "chinese"])
    parser.add_argument("--max_duration", type=float, default=30, help="Use only first N seconds of each file (default: 30, use 0 for full)")
    args = parser.parse_args()

    md = None if args.max_duration == 0 else args.max_duration
    train(args.data_dir, args.output, args.test_ratio, args.language, max_duration=md)
