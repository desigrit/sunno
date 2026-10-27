# Sunno

**Live captions for the conversation in front of you, running entirely on your own PC.**

![Sunno captioning a four person conversation, with each speaker labelled](docs/screenshot.png)

---

## Why this exists

I kept missing things.

Not the big things. I hear those. It is the small ones that go past: the end of a sentence when
someone turns their head, the name of the restaurant, the punchline everyone else laughs at. You
nod along and hope nobody asks a follow up question. If you have done that, you already know why
this app exists.

Sunno means "listen" in Hindi and Urdu.

Read about the [product decisions and model judgment](#product-decisions-and-model-judgment),
[how I worked with AI](#how-i-worked-with-ai), and [what I pushed back on](#what-i-pushed-back-on).

## What it does

Point it at a microphone and it writes down what people are saying, as they say it. Put it near
the table at dinner, beside you in a meeting, or pass a small mic to whoever is talking.

- **Captions as you speak.** Initial text appears while you are talking and is revised as each
  utterance ends. How quickly it arrives depends on the model and your machine.
- **Labels who is speaking.** A four way conversation reads like a conversation instead of one
  long paragraph. Rename anyone, and mark which one is you so your own lines step back.
- **Captions your PC's audio too,** which covers video calls, YouTube, podcasts, anything coming
  out of your speakers.
- **Handles accented speech well.** This was the whole reason for choosing Whisper large-v3 over
  faster alternatives. See [the engineering notes](docs/ENGINEERING.md#why-these-choices).
- **Shrinks to a strip** when you only want the words, and stays on top of whatever you are
  reading.
- **Runs without a graphics card.** Alongside the Whisper models there are two streaming ones,
  Zipformer and Kroko, built for machines with no GPU. Both are under 70 MB, offering a smaller
  download and an alternative to running Whisper on the processor.
- **Saves a conversation when you want one.** Press record and Sunno writes the audio and its
  transcript to your PC when you stop, with speaker labels and timestamps. Nothing is recorded
  unless you ask, the folder is not even created until the first save, and nothing is uploaded
  either way. See [Recording](#recording).
- Adjustable text size, always on top, and you can select and copy any part of the transcript.

<table>
<tr>
<td width="50%"><img alt="Sunno in compact mode, a small strip showing only the captions" src="docs/screenshot-compact.png"></td>
<td width="50%"><img alt="The speech model picker, listing seven models with their download size and expected delay" src="docs/screenshot-models.png"></td>
</tr>
<tr>
<td>Compact mode, for when you want the words and nothing else.</td>
<td>Pick a model. Each one shows its size and how far behind it will run on your PC.</td>
</tr>
</table>

## It runs on your PC. All of it.

This matters more than any feature.

Captioning apps usually stream your microphone to a company's servers. That means the
conversation at your dinner table, with your family in it, leaves your house. People who have not
thought about it will still feel it, and they are right to.

Sunno does the recognition on your own machine. No account, no sign in, no telemetry, no server
to send anything to. Once setup is done, turn off your Wi-Fi and it works exactly the same. The
people around you are not being uploaded anywhere, and you can tell them so honestly.

Sunno touches the network during setup and then never again. It downloads the speech model,
and on a PC with an NVIDIA graphics card the CUDA libraries that card needs. After that,
nothing.

Two things get written to disk, and both are yours: a pinned speaker's name and voice
fingerprint, so Sunno recognises them next time, and any recording you deliberately make. Full
details in [PRIVACY.md](PRIVACY.md).

## Product decisions and model judgment

I judge Sunno by whether I can follow a conversation while it is happening. That means
considering missed words, invented words, delay and distracting revisions together. A model
can produce a good transcript and still be a poor choice for live captions.

### Tradeoffs I made

**Accuracy has to fit the machine.** Accented speech was central to the original choice of
Whisper large-v3. Keeping it useful also meant offering smaller Whisper models and streaming
engines for PCs that cannot run it responsively. The model picker exposes the alternatives and
their estimated delay. A fixed "Recommended" label was removed because it could recommend a
model the app had already found too slow on that PC. The remaining tradeoff is real: choosing
a faster model can change what gets transcribed correctly. [Model-selection decisions](docs/CONTEXT.md#model-descriptions-say-what-a-model-is-never-whether-to-pick-it)

**Fast text and stable text are different goals.** The Windows Whisper path uses a quick
provisional pass, followed by a more expensive final pass over the completed utterance. That
keeps one speech model in memory while allowing an initial reading and a later correction.
It also adds computation and visible revisions. I treat how often a reader has to reread a
line as part of quality, alongside recognition errors. [Decoding choices](docs/ENGINEERING.md#why-these-choices)

**A hallucination filter can create its own failure.** Early filtering removed caption credits
Whisper sometimes invented, but also deleted legitimate speech such as "The book was translated
by Tolkien himself." Review exposed that the original passing examples did not cover this
kind of sentence. The filter was narrowed, and the regression set now checks both unwanted
credits and real speech that must survive. Those checks protect known cases; they do not
establish a live hallucination rate. [Filter regression cases](tests/test_hallucinations.py)

**Conversation context needs an expiry.** Recent recognized text helps with names and continuity,
but feeding model output back indefinitely can reinforce an error. The Whisper path uses a
bounded prompt that expires after silence, with automatic previous-text conditioning disabled.
That gives up some continuity to limit repeated mistakes. [Recognition and context handling](server/asr.py)

### How I worked with AI

AI coding assistants contributed implementation, tests and technical investigations. I used
the builds in conversations and while watching videos, described failures I could reproduce,
and made choices about what the experience should tolerate.

One recent example was a webinar with a single presenter being labeled as several people.
I asked whether speaker recognition could be improved or should be removed. The investigation
separated two systems I needed to judge independently: Whisper transcribes the words, while
CAM++ embeddings and matching logic assign the speakers. A transcription-model upgrade would
not by itself resolve that failure. [Speaker matching](server/speaker.py)

### What I pushed back on

For that speaker-labeling iteration, I rejected a dedicated single-speaker mode as a poor fit
for the changing videos I watch. I also did not want extra confirmations interrupting the
experience or saved voice profiles to become a prerequisite.

I asked for an Off / Automatic choice, an Experimental label on Automatic, and speaker labels
that could appear with the words and be corrected as more evidence arrived. These are choices
in ongoing Windows development, not a measured improvement in speaker accuracy. The cost is
that a provisional attribution can still be wrong. Testing has to measure both splitting one
person into several speakers and incorrectly combining different people.

### What I would measure next

I would compare the same speech across models, microphones and hardware, measuring time to
first useful text, time to stable text, word errors and revisions. Speaker attribution needs
its own evaluation on both single-presenter videos and conversations with real turn changes.
A decode timing on one machine does not establish the delay someone experiences from speech
to readable captions.

The next product decision is which configurations people can comfortably follow for a whole
conversation, including on battery. That needs listening sessions and reader feedback alongside
the existing regression tests.

## Recording

Press the record button beside the compact-mode button. Sunno writes the audio and its
transcript when you stop.

```
~/Sunno/Recordings/Recording/
  audio.m4a         AAC, 16 kHz mono, about 23 MB an hour
  transcript.json   speakers, timings, per word offsets
  transcript.txt    plain text with speaker labels and timestamps
```

Change the folder in Settings. A few things worth knowing:

- **Nothing is written until you press record.** The folder is not created until the first save,
  so an install that never records leaves no trace.
- **A recording ends when you stop it or close Sunno, and not before.** Pausing, swapping
  microphone, changing speech model and a dropped USB cable are all gaps inside one recording
  rather than the end of it.
- **An interrupted recording is finished on the next launch** rather than lost. Audio is appended
  as it arrives and each line is written as it is decoded, so a process that dies mid meeting
  costs you the sentence being spoken and nothing more.
- **The audio is the recogniser's own 16 kHz mono stream**, tapped before any conditioning. Right
  for speech and not archival, but it does mean the audio and the transcript can never disagree,
  because they are the same samples.

## Install

**Option A:**

<a href="https://get.microsoft.com/installer/download/9msv0v6jv8hk?referrer=appbadge&amp;cid=sunno-github"><picture><source media="(prefers-color-scheme: dark)" srcset="https://get.microsoft.com/images/en-us%20light.svg" width="200"><img alt="Get Sunno from the Microsoft Store" src="https://get.microsoft.com/images/en-us%20dark.svg" width="200"></picture></a>

**Option B: From source** - see [Building](#building) below.

On first run Sunno downloads a speech model. Whisper large-v3 is roughly 3 GB; the streaming
models are under 70 MB. On a PC with an NVIDIA graphics card it also fetches about 450 MB of
CUDA libraries, once, alongside the model. Those used to ship inside the installer, which
made it 828 MB larger for every PC including the ones that can never use them. Setup is the
only step that needs an internet connection.

It runs on any modern PC: with an NVIDIA GPU captions appear almost
immediately once those libraries are in place, and without one you can either use your
processor with a Whisper model, which works
but lags further behind, or pick one of the streaming models built for exactly that case.
The app measures your machine and tells you what to expect before you commit to a download.

**Looking for Sunno on a Mac?** It is a separate app, in
[desigrit/sunno-macos](https://github.com/desigrit/sunno-macos), with its own download.

## Building

Requires .NET 8, the Windows App SDK, and Python 3.12 from python.org. Not the Microsoft Store
Python build, which is sandboxed and unreliable at loading native CUDA DLLs.

```powershell
winget install --id Python.Python.3.12 --scope user
& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run the backend on its own, useful for testing without the desktop app:

```powershell
.\.venv\Scripts\python.exe -m server.app --list-devices
.\.venv\Scripts\python.exe -m server.app --device "Umik-1"
```

Build the installable package:

```powershell
.\packaging\build-msix.ps1
```

Tests are standalone scripts rather than pytest:

```powershell
.\.venv\Scripts\python.exe tests\test_biquad.py
.\.venv\Scripts\python.exe tests\test_hallucinations.py
```

## Honest about what it cannot do

No captioning system is perfect, and anyone who tells you otherwise is selling something.

- **Microphone placement matters more than the model.** Moving from a distant tabletop mic to a
  close talking one is worth roughly twice the accuracy, which is more than any model change
  available. There is a table of measured numbers in
  [the engineering notes](docs/ENGINEERING.md#microphone-matters-more-than-the-model).
- **Speaker labels are best effort.** Around two thirds of turns are correct on two speaker
  audio. Short turns like "yeah" or "okay, fine" are left unlabelled rather than guessed at,
  because embeddings need two to three seconds of speech to be dependable. Naming the people you
  talk to most is the single biggest improvement.
- **Overlapping speech degrades badly.** When two people talk at once, single channel
  recognition of any kind struggles. Separation models fast enough for real time are not
  practical yet on this hardware.
- **Across a noisy room it will drop words.**

It is a tool for catching more of what is said, not a court transcript. Used that way, it is
genuinely useful.

## How it fits together

```
microphone ──► resample 16 kHz ──► Silero VAD ──► Whisper large-v3 ──► WebSocket ──► WinUI app
                                                   (CUDA, or CPU)
```

A Python backend does capture and recognition. A WinUI 3 desktop app displays the results and
talks to it over a local WebSocket. They are separate processes on purpose, so the display could
later move to a handheld screen while this PC keeps doing the work.

| Directory | Contents |
|---|---|
| `server/` | Python backend: capture, VAD, recognition, speaker labelling |
| `app/` | WinUI 3 desktop app |
| `ui/` | Browser client, for the phone or handheld route |
| `packaging/` | Icon generation, CUDA trimming, MSIX build |
| `docs/CONTEXT.md` | Project context and accumulated learnings, for picking this up cold |
| `docs/ENGINEERING.md` | Why each decision was made, with the measurements behind it |

## Licence

MIT. See [LICENSE](LICENSE) and [THIRD-PARTY-NOTICES.md](THIRD-PARTY-NOTICES.md).

Sunno is free, and the source is public so you can read exactly what it does with your
microphone rather than taking my word for it. If you find something that contradicts
[PRIVACY.md](PRIVACY.md), that is a bug and I want to know about it.
