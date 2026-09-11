"""Build test video assets used by the baseline integration tests.

Generates deterministic MP4 files with real faces, animated mouths (to give
MediaPipe measurable mouth-motion evidence), and real synthesized speech from
the ffmpeg flite filter. Assets are written under test_assets/ and are
gitignored; integration tests skip when they are absent.

Usage:
    python scripts/build_test_assets.py
"""

import math
import os
import subprocess
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "test_assets")
os.makedirs(ASSETS, exist_ok=True)

# Face geometry (measured from the detected face in single_speaker.mp4).
FACE_H_W_RATIO = 366.0 / 303.0  # crop is portrait


def ensure_face_images() -> dict:
    """Locate the detected face crop used to build synthetic speaker videos.

    face_crop.png is extracted from test_assets/single_speaker.mp4 (a face the
    MediaPipe landmarker reliably detects). A horizontally flipped copy serves
    as the second speaker.

    Returns:
        Dict mapping short name to local path.
    """
    crop = os.path.join(ASSETS, "face_crop.png")
    if not os.path.isfile(crop):
        raise RuntimeError(
            f"Missing {crop}. Extract it from test_assets/single_speaker.mp4 "
            "first (detect() the first frame and crop the face bbox)."
        )
    return {"face": crop}


def flite_wav(text: str, out_wav: str, duration: float) -> str:
    """Synthesize speech with ffmpeg's flite filter.

    Args:
        text: Text to speak.
        out_wav: Output WAV path.
        duration: Desired duration in seconds (padded to exactly this length).

    Returns:
        Path to the generated WAV.
    """
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", f"flite=text='{text}':voice=slt",
        "-ar", "16000", "-ac", "1",
        "-af", "apad=pad_dur=30",
        "-t", str(duration),
        out_wav,
    ]
    subprocess.run(cmd, check=True)
    return out_wav


def concat_wavs(parts, out_wav):
    """Concatenate mono 16k WAV files including silent gaps."""
    inputs = []
    filter_parts = []
    index = 0
    for part in parts:
        wav, gap_seconds = part
        inputs += ["-i", wav]
        filter_parts.append(f"[{index}:a]")
        index += 1
        if gap_seconds > 0:
            inputs += ["-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono"]
            filter_parts.append(f"[{index}:a]")
            index += 1
    concat_input = "".join(filter_parts)
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        *inputs,
        "-filter_complex",
        f"{concat_input}concat=n={len(filter_parts)}:v=0:a=1[out]",
        "-map", "[out]", "-ar", "16000", "-ac", "1",
        out_wav,
    ]
    subprocess.run(cmd, check=True)
    return out_wav


def mouth_openness(t: float, seg_start: float, seg_end: float) -> float:
    """Return a 0..1 mouth openness for talking at time t."""
    if seg_start <= t < seg_end:
        return 0.5 + 0.5 * math.sin(2 * math.pi * 3.5 * (t - seg_start))
    return 0.0


MOUTH_REL_X = 0.61
MOUTH_REL_Y = 0.794
MOUTH_HALF_W = 0.15
MOUTH_OPEN_H = 0.05
MOUTH_CLOSED_H = 0.003


def draw_face_with_mouth(
    canvas: np.ndarray,
    face_img: np.ndarray,
    x: int,
    y: int,
    target_w: int,
    target_h: int,
    openness: float,
    mirrored: bool = False,
) -> None:
    """Paste a face and draw a synthetic mouth whose height tracks openness.

    The mouth is drawn at the measured mouth location of the source face crop
    (0.61 / 0.79 of the face box) so MediaPipe's lip landmarks move. For a
    horizontally mirrored face the mouth x-coordinate is mirrored too.

    Args:
        canvas: BGR canvas to draw on.
        face_img: BGR face image.
        x, y: Top-left corner of the face box in canvas coords.
        target_w, target_h: Face box size.
        openness: Mouth openness in [0, 1].
        mirrored: True when face_img was flipped horizontally.
    """
    face = cv2.resize(
        face_img,
        (target_w, target_h),
        interpolation=cv2.INTER_AREA,
    )
    canvas[y:y + target_h, x:x + target_w] = face

    mouth_rel_x = (1.0 - MOUTH_REL_X) if mirrored else MOUTH_REL_X
    mouth_cx = x + int(target_w * mouth_rel_x)
    mouth_cy = y + int(target_h * MOUTH_REL_Y)
    mouth_half_w = int(target_w * MOUTH_HALF_W)
    mouth_half_h = int(target_h * (MOUTH_CLOSED_H + MOUTH_OPEN_H * openness))

    if mouth_half_h < 2:
        cv2.line(
            canvas,
            (mouth_cx - mouth_half_w, mouth_cy),
            (mouth_cx + mouth_half_w, mouth_cy),
            (0, 0, 0), 2,
        )
    else:
        cv2.ellipse(
            canvas,
            (mouth_cx, mouth_cy),
            (mouth_half_w, mouth_half_h),
            0, 0, 360, (0, 0, 0), -1,
        )


def write_video(path: str, frames, fps: int, wav: str):
    """Write frames + audio to an MP4 (h264 + aac)."""
    height, width = frames[0].shape[:2]
    temp_video = path + ".raw.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(temp_video, fourcc, fps, (width, height))
    for frame in frames:
        writer.write(frame)
    writer.release()

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", temp_video,
        "-i", wav,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        "-shortest",
        path,
    ]
    subprocess.run(cmd, check=True)
    os.remove(temp_video)
    print(f"Wrote {path}")


def silent_wav(duration: float, out_wav: str) -> str:
    """Create a silent mono 16k WAV of a given duration."""
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "lavfi", "-i", "anullsrc=r=16000:cl=mono",
        "-t", str(duration),
        out_wav,
    ]
    subprocess.run(cmd, check=True)
    return out_wav


SEG_A = "This speaker is the left person talking right now."
SEG_B = "Now the right person is asking a clarifying question."
SEG_A2 = "The left person agrees and continues the conversation."


def build_single_speaker(faces):
    """single_speaker_speech.mp4: one face, speaking the whole time."""
    w, h, fps, duration = 1920, 1080, 30, 6.0
    n_frames = int(fps * duration)
    face_img = cv2.imread(faces["face"])
    fw, fh = 560, int(560 * FACE_H_W_RATIO)
    frames = []
    for i in range(n_frames):
        t = i / fps
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        draw_face_with_mouth(canvas, face_img, 720, 220, fw, fh, mouth_openness(t, 0.0, duration))
        frames.append(canvas)
    wav = flite_wav(SEG_A, os.path.join(ASSETS, "single_speech.wav"), duration)
    write_video(os.path.join(ASSETS, "single_speaker_speech.mp4"), frames, fps, wav)
    os.remove(wav)


def build_two_speakers(faces):
    """two_speakers.mp4: two faces; A talks, then B, then A again."""
    w, h, fps, duration = 1280, 720, 30, 9.0
    seg_length = 3.0
    n_frames = int(fps * duration)
    face_img = cv2.imread(faces["face"])
    face_flip = cv2.flip(face_img, 1)
    fw, fh = 440, int(440 * FACE_H_W_RATIO)

    frames = []
    for i in range(n_frames):
        t = i / fps
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        right_speaking = seg_length <= t < 2 * seg_length
        left_open = mouth_openness(t, 0.0, seg_length) if t < seg_length else (
            mouth_openness(t, 2 * seg_length, 3 * seg_length) if t >= 2 * seg_length else 0.0
        )
        right_open = (
            mouth_openness(t, seg_length, 2 * seg_length)
            if right_speaking
            else 0.0
        )
        draw_face_with_mouth(
            canvas, face_img, 120, 100, fw, fh, left_open,
        )
        draw_face_with_mouth(
            canvas, face_flip, 720, 100, fw, fh, right_open,
            mirrored=True,
        )
        frames.append(canvas)

    parts = [
        (flite_wav(SEG_A, os.path.join(ASSETS, "_a.wav"), seg_length), 0.0),
        (flite_wav(SEG_B, os.path.join(ASSETS, "_b.wav"), seg_length), 0.0),
        (flite_wav(SEG_A2, os.path.join(ASSETS, "_a2.wav"), seg_length), 0.0),
    ]
    wav = os.path.join(ASSETS, "two_speakers.wav")
    concat_wavs(parts, wav)
    write_video(os.path.join(ASSETS, "two_speakers.mp4"), frames, fps, wav)
    for temp in ("_a.wav", "_b.wav", "_a2.wav", "two_speakers.wav"):
        p = os.path.join(ASSETS, temp)
        if os.path.isfile(p):
            os.remove(p)


def build_silence(faces):
    """silence_speaker.mp4: static face with silent audio (no speech)."""
    w, h, fps, duration = 1920, 1080, 30, 6.0
    n_frames = int(fps * duration)
    face_img = cv2.imread(faces["face"])
    fw, fh = 560, int(560 * FACE_H_W_RATIO)
    frames = []
    for i in range(n_frames):
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        draw_face_with_mouth(canvas, face_img, 720, 220, fw, fh, 0.0)
        frames.append(canvas)
    wav = silent_wav(duration, os.path.join(ASSETS, "silence.wav"))
    write_video(os.path.join(ASSETS, "silence_speaker.mp4"), frames, fps, wav)
    os.remove(wav)


def build_face_loss(faces):
    """face_loss.mp4: face talks, disappears for 2s, reappears talking."""
    w, h, fps = 1920, 1080, 30
    duration = 8.0
    n_frames = int(fps * duration)
    face_img = cv2.imread(faces["face"])
    fw, fh = 560, int(560 * FACE_H_W_RATIO)
    frames = []
    for i in range(n_frames):
        t = i / fps
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        visible = t < 3.0 or t >= 5.0
        if visible:
            draw_face_with_mouth(canvas, face_img, 720, 220, fw, fh, mouth_openness(t, 0.0, duration))
        else:
            cv2.putText(canvas, "no face", (860, 540), cv2.FONT_HERSHEY_SIMPLEX, 2, (80, 80, 80), 3)
        frames.append(canvas)
    wav = flite_wav(SEG_A + " " + SEG_A2, os.path.join(ASSETS, "face_loss_speech.wav"), duration)
    write_video(os.path.join(ASSETS, "face_loss.mp4"), frames, fps, wav)
    os.remove(wav)


def build_off_screen(faces):
    """off_screen_speech.mp4: two static faces, speech from an offscreen actor."""
    w, h, fps, duration = 1280, 720, 30, 6.0
    n_frames = int(fps * duration)
    face_img = cv2.imread(faces["face"])
    face_flip = cv2.flip(face_img, 1)
    fw, fh = 440, int(440 * FACE_H_W_RATIO)
    frames = []
    for i in range(n_frames):
        canvas = np.zeros((h, w, 3), dtype=np.uint8)
        draw_face_with_mouth(canvas, face_img, 120, 100, fw, fh, 0.0)
        draw_face_with_mouth(canvas, face_flip, 720, 100, fw, fh, 0.0, mirrored=True)
        frames.append(canvas)
    wav = flite_wav(SEG_A, os.path.join(ASSETS, "off_screen.wav"), duration)
    write_video(os.path.join(ASSETS, "off_screen_speech.mp4"), frames, fps, wav)
    os.remove(wav)


def build_no_face():
    """no_face_speech.mp4: no faces at all, with speech audio."""
    w, h, fps, duration = 640, 360, 30, 6.0
    n_frames = int(fps * duration)
    frames = []
    for i in range(n_frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        x = int(w * i / n_frames)
        cv2.rectangle(frame, (x, 100), (x + 80, 220), (0, 150, 255), -1)
        frames.append(frame)
    wav = flite_wav(SEG_A, os.path.join(ASSETS, "no_face_speech.wav"), duration)
    write_video(os.path.join(ASSETS, "no_face_speech.mp4"), frames, fps, wav)
    os.remove(wav)


def main():
    faces = ensure_face_images()
    build_single_speaker(faces)
    build_two_speakers(faces)
    build_silence(faces)
    build_face_loss(faces)
    build_off_screen(faces)
    build_no_face()
    print("All test assets built.")


if __name__ == "__main__":
    sys.exit(main())