"""
Voice I/O. Kept optional and isolated: if speech_recognition / pyttsx3 /
pyaudio aren't installed or no mic is available, the rest of the assistant
still works fine in text mode.
"""
import sys

_sr = None
_tts_engine = None


def voice_available() -> bool:
    try:
        import speech_recognition  # noqa: F401
        import pyttsx3  # noqa: F401
        return True
    except ImportError:
        return False


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


def speak(text: str):
    """Speak text aloud. Falls back to printing if TTS isn't available."""
    try:
        engine = _get_tts_engine()
        engine.say(text)
        engine.runAndWait()
    except Exception:
        # Never let a TTS hiccup take down the assistant.
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
