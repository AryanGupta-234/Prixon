"""Voice I/O (spec sections 42-43).

TTS tries streaming neural speech first (edge-tts + ffmpeg decode +
sounddevice playback) -- audio starts as it's synthesized rather than
waiting for the whole clip to render, which is the streaming principle
spec section 43 asks for, and edge-tts's neural voices are the "high-quality
neural voices" spec section 42 calls out. It falls back to the fully
offline pyttsx3 engine if edge-tts, ffmpeg, or sounddevice aren't
available/working, and falls back to doing nothing (never raising) if
neither works -- same "never let a TTS hiccup take down the assistant"
contract this file always had, just with a better primary path now. This
mirrors how brain/router.py already prefers a better option and falls back
to an always-available one rather than the app just breaking when the
better option isn't there.

All of the heavy imports (edge_tts, sounddevice, numpy, pyttsx3,
speech_recognition) are lazy/inside functions, and voice_available() and
the two _*_available() checks never raise ImportError outward -- so
`import voice` itself must always succeed even with zero optional voice
dependencies installed, since main.py imports this module unconditionally
regardless of whether --voice was passed.

STT is unchanged from before: speech_recognition + Google's free
recognizer, lazy-loaded the same way.
"""
from __future__ import annotations

import queue
import shutil
import subprocess
import sys
import threading

_sr = None
_tts_engine = None
_streaming_tts_checked = None  # None = not checked yet; True/False after first check


def _pyttsx3_available() -> bool:
    try:
        import pyttsx3  # noqa: F401
        return True
    except ImportError:
        return False


def _streaming_tts_available() -> bool:
    """edge-tts + sounddevice + numpy are pip packages, but the actual MP3
    decode step shells out to the ffmpeg binary, which is NOT pip-installable
    -- so this must also check shutil.which('ffmpeg') before claiming the
    streaming path works, or speak() would only discover the missing binary
    mid-utterance. Cached after the first check so a missing ffmpeg doesn't
    get re-probed via shutil.which on every single speak() call."""
    global _streaming_tts_checked
    if _streaming_tts_checked is not None:
        return _streaming_tts_checked
    try:
        import edge_tts  # noqa: F401
        import sounddevice  # noqa: F401
        import numpy  # noqa: F401
        _streaming_tts_checked = shutil.which("ffmpeg") is not None
    except Exception:
        # Not just ImportError: sounddevice raises OSError at import time if
        # the underlying PortAudio system library isn't installed, which is
        # a very likely first-run state on a fresh machine, not a bug.
        _streaming_tts_checked = False
    return _streaming_tts_checked


def voice_available() -> bool:
    """True if STT works AND at least one TTS backend works. speak() itself
    picks which TTS backend at call time; this is just the gate main.py
    uses to decide whether --voice mode makes sense at all."""
    try:
        import speech_recognition  # noqa: F401
        stt_ok = True
    except ImportError:
        stt_ok = False
    return stt_ok and (_pyttsx3_available() or _streaming_tts_available())


def tts_backend() -> str:
    """For diagnostics.py's health report -- which TTS path is actually
    live right now, not just 'voice works or doesn't.'"""
    if _streaming_tts_available():
        return "streaming (edge-tts)"
    if _pyttsx3_available():
        return "offline (pyttsx3)"
    return "unavailable"


def _get_recognizer():
    global _sr
    if _sr is None:
        import speech_recognition as sr
        _sr = sr
    return _sr


def _get_tts_engine():
    global _tts_engine
    if _tts_engine is None:
        import pyttsx3
        import config
        _tts_engine = pyttsx3.init()
        _tts_engine.setProperty("rate", config.TTS_RATE)
        _tts_engine.setProperty("volume", config.TTS_VOLUME)
    return _tts_engine


def _speak_pyttsx3(text: str):
    engine = _get_tts_engine()
    engine.say(text)
    engine.runAndWait()


def _audio_player_thread(pcm_queue: "queue.Queue", sample_rate: int, errors: list):
    try:
        import sounddevice as sd
        stream = sd.OutputStream(samplerate=sample_rate, channels=1, dtype="int16")
        stream.start()
        try:
            while True:
                chunk = pcm_queue.get()
                if chunk is None:
                    break
                stream.write(chunk)
        finally:
            stream.stop()
            stream.close()
    except Exception as exc:  # noqa: BLE001
        errors.append(exc)


def _ffmpeg_reader_thread(proc, pcm_queue: "queue.Queue", errors: list):
    try:
        import numpy as np
        while True:
            data = proc.stdout.read(4096)
            if not data:
                break
            pcm_queue.put(np.frombuffer(data, dtype=np.int16))
    except Exception as exc:  # noqa: BLE001
        errors.append(exc)
    finally:
        pcm_queue.put(None)


async def _speak_stream_async(text: str, voice: str, rate: str, pitch: str, sample_rate: int):
    import edge_tts

    ffmpeg_proc = subprocess.Popen(
        ["ffmpeg", "-loglevel", "quiet", "-f", "mp3", "-i", "pipe:0",
         "-f", "s16le", "-ar", str(sample_rate), "-ac", "1", "pipe:1"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
    )
    pcm_queue: "queue.Queue" = queue.Queue()
    errors: list = []
    reader = threading.Thread(target=_ffmpeg_reader_thread, args=(ffmpeg_proc, pcm_queue, errors))
    player = threading.Thread(target=_audio_player_thread, args=(pcm_queue, sample_rate, errors))
    reader.start()
    player.start()

    try:
        communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                ffmpeg_proc.stdin.write(chunk["data"])
                ffmpeg_proc.stdin.flush()
    finally:
        try:
            ffmpeg_proc.stdin.close()
        except Exception:
            pass
        reader.join(timeout=10)
        player.join(timeout=10)
        ffmpeg_proc.wait(timeout=10)

    if errors:
        raise errors[0]  # let speak() catch this and fall back to pyttsx3


def _speak_streaming(text: str):
    import asyncio
    import config
    asyncio.run(_speak_stream_async(
        text, voice=config.TTS_VOICE, rate=config.TTS_STREAM_RATE,
        pitch=config.TTS_STREAM_PITCH, sample_rate=config.TTS_STREAM_SAMPLE_RATE,
    ))


def speak(text: str):
    """Speak text aloud. Streaming neural TTS first, offline pyttsx3 second,
    silent no-op last -- never lets a TTS failure take down the assistant."""
    if _streaming_tts_available():
        try:
            _speak_streaming(text)
            return
        except Exception:
            pass  # fall through to the offline engine below
    try:
        _speak_pyttsx3(text)
    except Exception:
        pass


def listen_once(timeout_seconds: int = None, phrase_time_limit: int = None) -> str:
    """
    Records one utterance from the default microphone and transcribes it.
    Returns "" if nothing usable was heard.
    """
    import config
    sr = _get_recognizer()
    timeout_seconds = timeout_seconds or config.MIC_TIMEOUT_SECONDS
    phrase_time_limit = phrase_time_limit or config.MIC_PHRASE_TIME_LIMIT

    recognizer = sr.Recognizer()
    try:
        with sr.Microphone() as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.4)
            print("Listening...")
            audio = recognizer.listen(
                source, timeout=timeout_seconds, phrase_time_limit=phrase_time_limit
            )
    except sr.WaitTimeoutError:
        return ""
    except OSError as e:
        print(f"Microphone error: {e}", file=sys.stderr)
        return ""

    try:
        return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return ""
    except sr.RequestError as e:
        print(f"Speech recognition service error: {e}", file=sys.stderr)
        return ""
