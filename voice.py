"""
Voice I/O. Kept optional and isolated: if speech_recognition / pyttsx3 /
pyaudio aren't installed or no mic is available, the rest of the assistant
still works fine in text mode.
"""
import sys

_sr = None
_tts_engine = None

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
### **The Last Train Home**

It was almost midnight when I realized I had missed the last train.

The station was completely empty.

Well… almost empty.

At the far end of the platform, an old man was sitting beneath a flickering light, calmly drinking tea as if it were the middle of the afternoon.

I walked over and asked, “Excuse me… is there another train coming?”

He looked at his watch.

Then he looked at me.

And smiled.

“Eventually.”

I laughed nervously. “Eventually?”

He nodded. “That depends on where you're trying to go.”

There was something strangely comforting about the way he said it.

So I sat down beside him.

For a few minutes, neither of us spoke. We just listened to the rain tapping against the metal roof.

Then—

**CLANG.**

I jumped.

A train had appeared on the opposite platform.

No announcement.

No headlights.

Just an old, silver train sitting silently in the darkness.

I stared at it.

“Was that there before?”

The old man smiled again.

“No.”

I looked at him.

“Then where did it come from?”

He took another sip of tea.

“That's the interesting part.”

Suddenly, the train doors opened.

A warm golden light spilled onto the platform.

And from inside came the sound of music.

Not loud.

Not frightening.

Just… beautiful.

I stood up.

The old man remained seated.

“Aren't you coming?”

He shook his head.

“I already arrived.”

I frowned. “Arrived where?”

He pointed toward the train.

“Home.”

For some reason, those words hit me harder than they should have.

I stepped toward the doors.

Then I stopped.

“Wait.”

I turned around.

The old man was gone.

His cup was still sitting on the bench.

Steam was still rising from it.

I looked at the train.

The doors began to close.

“Hey!”

I ran.

Just before they shut, I slipped inside.

The train started moving.

Outside the window, the station disappeared into darkness.

And then I noticed something strange.

Every passenger in the carriage was looking at me.

Smiling.

Like they had been waiting for me.

I slowly sat down.

My phone suddenly buzzed.

One new message.

From an unknown number.

It said:

**“You finally made it home.”**

I stared at the screen.

Then I laughed.

Because underneath the message was a photo.

A photo of me…

sitting on that empty platform.

Taken from behind.

And in the corner of the picture—

the old man was standing there.

Smiling.

Watching me leave.

I looked up from my phone.

The carriage was empty.

Completely empty.

Except for one seat across from me.

And on that seat…

was a cup of hot tea.

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

 
