import edge_tts
import asyncio
import subprocess
import sounddevice as sd
import numpy as np
import threading
import queue

VOICE = "en-US-AvaNeural"
SAMPLE_RATE = 24000  # edge-tts default output rate

test_text = """
Good morning, sir. All systems are online and running smoothly.
Wait — I'm detecting a critical anomaly in the power grid, this needs your attention immediately!
Hmm, that's strange. I've never seen a reading quite like this before.
Oh, don't worry about it, I'm sure it's nothing serious. Really.
Honestly, I told you this would happen eventually, didn't I?
I'm so glad that worked out. You should be proud of yourself.
And... we're back to normal operations. Standing by for your next command.
"""

def audio_player_thread(pcm_queue, stop_event):
    """Pulls decoded PCM chunks off the queue and plays them as they arrive."""
    stream = sd.OutputStream(samplerate=SAMPLE_RATE, channels=1, dtype="int16")
    stream.start()
    while not stop_event.is_set() or not pcm_queue.empty():
        try:
            chunk = pcm_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        if chunk is None:
            break
        stream.write(chunk)
    stream.stop()
    stream.close()

def ffmpeg_reader_thread(proc, pcm_queue):
    """Reads decoded PCM from ffmpeg's stdout and pushes to the play queue."""
    while True:
        data = proc.stdout.read(4096)
        if not data:
            break
        pcm_array = np.frombuffer(data, dtype=np.int16)
        pcm_queue.put(pcm_array)
    pcm_queue.put(None)

async def speak_stream(text, voice=VOICE, rate="+5%", pitch="+0Hz"):
    ffmpeg_proc = subprocess.Popen(
        [
            "ffmpeg", "-loglevel", "quiet",
            "-f", "mp3", "-i", "pipe:0",
            "-f", "s16le", "-ar", str(SAMPLE_RATE), "-ac", "1",
            "pipe:1",
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )

    pcm_queue = queue.Queue()
    stop_event = threading.Event()

    reader_thread = threading.Thread(target=ffmpeg_reader_thread, args=(ffmpeg_proc, pcm_queue))
    player_thread = threading.Thread(target=audio_player_thread, args=(pcm_queue, stop_event))
    reader_thread.start()
    player_thread.start()

    communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            ffmpeg_proc.stdin.write(chunk["data"])
            ffmpeg_proc.stdin.flush()

    ffmpeg_proc.stdin.close()
    reader_thread.join()
    stop_event.set()
    player_thread.join()
    ffmpeg_proc.wait()

asyncio.run(speak_stream(test_text))

 
