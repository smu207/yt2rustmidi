#!/usr/bin/env python3
"""
yt2midi.py — 유튜브 노래를 악기별로 분리하고, 각 악기를 MIDI 파일로 변환합니다.

파이프라인
  1) yt-dlp      : 유튜브 오디오 다운로드 (WAV)
  2) Demucs      : 보컬 / 드럼 / 베이스 / 기타 / 피아노 / 기타(other) 분리
  3) Basic Pitch : 음정 있는 악기(보컬·베이스·기타·피아노·other) → MIDI
     드럼        : 주파수 대역별 온셋 검출 → GM 드럼 MIDI (킥/스네어/하이햇)
  4) 악기별 .mid + 전체를 합친 multitrack .mid 저장

사용 예
  python yt2midi.py "https://www.youtube.com/watch?v=XXXXXXXXXXX"
  python yt2midi.py song.mp3                      # 로컬 파일도 가능
  python yt2midi.py URL --model htdemucs          # 4트랙 모델(보컬/드럼/베이스/other)
  python yt2midi.py URL --device cuda --out result
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np

# ----------------------------------------------------------------------------
# 악기별 설정 (GM 프로그램 번호, 음역, 단선율 여부, Basic Pitch 파라미터)
# ----------------------------------------------------------------------------
STEM_CONFIG = {
    "vocals": dict(name="Vocals", program=53, min_freq=80, max_freq=1200,
                   mono=True, onset=0.5, frame=0.3, min_len_ms=100),
    "bass":   dict(name="Bass", program=33, min_freq=30, max_freq=400,
                   mono=True, onset=0.5, frame=0.3, min_len_ms=80),
    "guitar": dict(name="Guitar", program=27, min_freq=70, max_freq=1500,
                   mono=False, onset=0.5, frame=0.3, min_len_ms=80),
    "piano":  dict(name="Piano", program=0, min_freq=27, max_freq=4200,
                   mono=False, onset=0.5, frame=0.3, min_len_ms=60),
    "other":  dict(name="Other", program=48, min_freq=50, max_freq=3000,
                   mono=False, onset=0.6, frame=0.4, min_len_ms=100),
}

# 드럼: GM 드럼 노트 번호와 검출할 주파수 대역(Hz)
DRUM_BANDS = {
    36: ("Kick",   30,   150),
    38: ("Snare",  180,  2500),
    42: ("Hi-hat", 6000, 16000),
}

SILENCE_RMS = 1e-3  # 이보다 조용한 스템은 비어 있는 것으로 보고 건너뜀


def log(msg):
    print(f"[yt2midi] {msg}", flush=True)


def safe_name(text, maxlen=80):
    text = re.sub(r'[\\/:*?"<>|]+', "_", text).strip()
    return (text or "song")[:maxlen]


# ----------------------------------------------------------------------------
# 1) 다운로드
# ----------------------------------------------------------------------------
def download_audio(url, workdir: Path):
    import yt_dlp

    workdir.mkdir(parents=True, exist_ok=True)
    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(workdir / "%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": False,
        "postprocessors": [
            {"key": "FFmpegExtractAudio", "preferredcodec": "wav"}
        ],
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        wav_path = Path(ydl.prepare_filename(info)).with_suffix(".wav")
    title = info.get("title") or info.get("id") or "song"
    if not wav_path.exists():
        raise FileNotFoundError(f"다운로드한 WAV를 찾을 수 없습니다: {wav_path}")
    return wav_path, title


# ----------------------------------------------------------------------------
# 2) 음원 분리 (Demucs)
# ----------------------------------------------------------------------------
def separate(audio_path: Path, stem_dir: Path, model_name: str, device: str):
    import librosa
    import soundfile as sf
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    stem_dir.mkdir(parents=True, exist_ok=True)
    log(f"Demucs 모델 로드: {model_name} (device={device})")
    model = get_model(model_name)
    model.eval()

    sr = model.samplerate
    wav, _ = librosa.load(str(audio_path), sr=sr, mono=False)
    if wav.ndim == 1:
        wav = np.stack([wav, wav])
    wav = torch.tensor(wav, dtype=torch.float32)

    ref = wav.mean(0)
    mean, std = ref.mean(), ref.std() + 1e-8
    wav = (wav - mean) / std

    log("음원 분리 중... (CPU라면 곡당 수 분 걸릴 수 있어요)")
    with torch.no_grad():
        sources = apply_model(model, wav[None], device=device, shifts=1,
                              split=True, overlap=0.25, progress=True)[0]
    sources = sources * std + mean

    stems = {}
    for name, src in zip(model.sources, sources):
        out = stem_dir / f"{name}.wav"
        sf.write(str(out), src.cpu().numpy().T, sr)
        stems[name] = out
        log(f"  스템 저장: {out.name}")
    return stems


def is_silent(path: Path):
    import soundfile as sf
    data, _ = sf.read(str(path), always_2d=True)
    return float(np.sqrt(np.mean(data ** 2))) < SILENCE_RMS


# ----------------------------------------------------------------------------
# 3-a) 음정 악기 → MIDI (Basic Pitch)
# ----------------------------------------------------------------------------
def make_monophonic(notes, min_dur=0.05):
    """겹치는 음을 정리해 한 번에 한 음만 남김 (보컬/베이스용)."""
    notes = sorted(notes, key=lambda n: (n.start, -n.velocity))
    result = []
    for n in notes:
        if result and n.start < result[-1].end:
            prev = result[-1]
            if abs(n.start - prev.start) < 0.03:
                # 거의 동시에 시작 → 더 크게 검출된 음만 유지
                if n.velocity > prev.velocity:
                    result[-1] = n
                continue
            prev.end = n.start  # 이전 음을 새 음 시작 지점에서 자름
        result.append(n)
    return [n for n in result if n.end - n.start >= min_dur]


def transcribe_pitched(stem_path: Path, cfg):
    import pretty_midi
    from basic_pitch import ICASSP_2022_MODEL_PATH
    from basic_pitch.inference import predict

    _, midi_data, _ = predict(
        str(stem_path),
        ICASSP_2022_MODEL_PATH,
        onset_threshold=cfg["onset"],
        frame_threshold=cfg["frame"],
        minimum_note_length=cfg["min_len_ms"],
        minimum_frequency=cfg["min_freq"],
        maximum_frequency=cfg["max_freq"],
        melodia_trick=True,
    )
    notes = [pretty_midi.Note(n.velocity, n.pitch, n.start, n.end)
             for inst in midi_data.instruments for n in inst.notes]
    if cfg["mono"]:
        notes = make_monophonic(notes)

    inst = pretty_midi.Instrument(program=cfg["program"], name=cfg["name"])
    inst.notes = sorted(notes, key=lambda n: n.start)
    return inst


# ----------------------------------------------------------------------------
# 3-b) 드럼 → MIDI (대역별 온셋 검출)
# ----------------------------------------------------------------------------
def transcribe_drums(stem_path: Path):
    import librosa
    import pretty_midi
    from scipy.signal import butter, sosfiltfilt

    y, sr = librosa.load(str(stem_path), sr=44100, mono=True)
    inst = pretty_midi.Instrument(program=0, is_drum=True, name="Drums")
    hop = 256

    for pitch, (label, lo, hi) in DRUM_BANDS.items():
        hi = min(hi, sr / 2 - 100)
        sos = butter(4, [lo, hi], btype="bandpass", fs=sr, output="sos")
        band = sosfiltfilt(sos, y)

        env = librosa.onset.onset_strength(y=band, sr=sr, hop_length=hop)
        if env.max() <= 0:
            continue
        frames = librosa.onset.onset_detect(
            onset_envelope=env, sr=sr, hop_length=hop,
            delta=0.07 * env.max(), wait=int(0.05 * sr / hop), units="frames",
        )
        times = librosa.frames_to_time(frames, sr=sr, hop_length=hop)
        peak = np.percentile(env[frames], 95) if len(frames) else 1.0

        for f, t in zip(frames, times):
            vel = int(np.clip(40 + 87 * env[f] / (peak + 1e-8), 30, 127))
            inst.notes.append(pretty_midi.Note(vel, pitch, float(t), float(t) + 0.1))
        log(f"  드럼 {label}: {len(frames)}타")

    inst.notes.sort(key=lambda n: n.start)
    return inst


# ----------------------------------------------------------------------------
# 템포 추정 (DAW에서 마디선이 대략 맞도록)
# ----------------------------------------------------------------------------
def estimate_tempo(audio_path: Path):
    import librosa
    try:
        y, sr = librosa.load(str(audio_path), sr=22050, mono=True, duration=120)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = float(np.atleast_1d(tempo)[0])
        return bpm if 40 < bpm < 240 else 120.0
    except Exception:
        return 120.0


def save_midi(instruments, path: Path, bpm: float):
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    pm.instruments.extend(instruments)
    pm.write(str(path))


# ----------------------------------------------------------------------------
# 메인
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="유튜브 노래 → 악기별 MIDI")
    ap.add_argument("source", help="유튜브 URL 또는 로컬 오디오 파일 경로")
    ap.add_argument("--out", default="output", help="결과 폴더 (기본: output)")
    ap.add_argument("--model", default="htdemucs_6s",
                    help="Demucs 모델: htdemucs_6s(6트랙, 기본) / htdemucs / htdemucs_ft")
    ap.add_argument("--device", default=None, help="cuda / cpu (기본: 자동)")
    ap.add_argument("--keep-silent", action="store_true",
                    help="거의 무음인 스템도 MIDI로 변환")
    args = ap.parse_args()

    if args.device is None:
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    out_root = Path(args.out)
    src = Path(args.source)

    # 1) 입력 준비
    if src.exists():
        audio_path, title = src, src.stem
        log(f"로컬 파일 사용: {src}")
    else:
        log("유튜브에서 오디오 다운로드 중...")
        audio_path, title = download_audio(args.source, out_root / "_downloads")

    song_dir = out_root / safe_name(title)
    stem_dir, midi_dir = song_dir / "stems", song_dir / "midi"
    midi_dir.mkdir(parents=True, exist_ok=True)
    log(f"곡: {title}")

    # 2) 분리
    stems = separate(audio_path, stem_dir, args.model, args.device)

    # 3) 변환
    bpm = estimate_tempo(audio_path)
    log(f"추정 템포: {bpm:.1f} BPM")

    all_insts = []
    for name, path in stems.items():
        if not args.keep_silent and is_silent(path):
            log(f"  {name}: 거의 무음이라 건너뜀")
            continue
        log(f"MIDI 변환 중: {name}")
        try:
            if name == "drums":
                inst = transcribe_drums(path)
            else:
                cfg = STEM_CONFIG.get(name, STEM_CONFIG["other"])
                inst = transcribe_pitched(path, cfg)
        except Exception as e:
            log(f"  {name} 변환 실패: {e}")
            continue

        if not inst.notes:
            log(f"  {name}: 검출된 음이 없음")
            continue
        save_midi([inst], midi_dir / f"{name}.mid", bpm)
        all_insts.append(inst)
        log(f"  → {name}.mid ({len(inst.notes)}개 음)")

    # 4) 전체 합본
    if all_insts:
        save_midi(all_insts, song_dir / "full_multitrack.mid", bpm)
        log(f"합본 저장: {song_dir / 'full_multitrack.mid'}")

    log(f"완료! 결과 폴더: {song_dir.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(1)
