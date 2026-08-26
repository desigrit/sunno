# Sunno: project context and accumulated learnings

Everything a person or a machine needs to pick this project up cold. It exists because the
reasoning behind this app lives in three places that do not travel together: the code
comments, the commit messages, and conversations that are lost when a session ends. This is
the third of those, written down.

Read it if you are starting on a new machine, joining the project, or returning after a gap.
It is organised by theme rather than by date, because a decision is easier to use than a
chronology. Where a number appears it was measured, and the thing that measured it is named
so it can be re-run rather than believed.

Companion documents, and what each is for:

| File | Purpose |
|---|---|
| `README.md` | What Sunno is, for someone deciding whether to install it |
| `docs/ENGINEERING.md` | Why the pipeline looks the way it does, with the measurements |
| `docs/ARM-PORT.md` | The Windows on ARM port, unfinished, six items open |
| `THIRD-PARTY-NOTICES.md` | Licences, including the one that is a known gap |
| `PRIVACY.md` | What is stored, what is sent, and what the diagnostics report contains |
| This file | The context none of the above has a natural home for |

---

## 1. What this is, and who it is for

Sunno is a fully offline live-captioning app for Windows, on the Microsoft Store. It listens
to a microphone or to the PC's own audio output and writes down what it hears, in a window
you can park beside a conversation.

**The author is hard of hearing and uses it daily.** That is the single most load-bearing fact
about the project and it decides arguments that would otherwise be close. Sunno means "listen"
in Hindi and Urdu.

It is an accessibility tool rather than a demo, and the difference shows up in the failure
modes that are considered unacceptable:

- A caption that arrives too late to follow the conversation is a failure, even if it is correct.
- A caption that invents a word is worse than one that arrives a moment later, because the
  person reading it cannot check it against what they heard.
- A transcript that silently stops is the worst failure of all, because it looks like a quiet
  room. Several bugs in this history are variations on that one theme.
- Anything that makes the app harder to trust in front of other people is a product problem,
  not a polish problem.

### The two audiences for the privacy promise

`PRIVACY.md` is explicit that it is written for two groups. The first is the person who
installed Sunno. The second is everyone whose voice it captures: people at the table, in the
meeting, on the other end of a call, who installed nothing and agreed to nothing. The second
group is the one the policy is really for.

This is why "runs entirely on your machine" is not a feature bullet but a constraint that
outranks features. The user has to be able to tell a room, honestly, that nothing is being
uploaded.

---

## 2. The promises, and what they cost

These are non-negotiable. Each has already cost something to keep.

**Fully offline after the model download.** There is exactly one network operation in the
entire application: fetching the model on first run. Turn off the Wi-Fi afterwards and it
works identically. No account, no telemetry, no crash reporting service.

**No transcript text on disk, ever.** This was not free. `backend.log` originally recorded
every finalised caption, and on the development machine it accumulated 316 lines of real
conversation including a work meeting. Commit `42b44f5` stopped it. The backend now prints a
numeric id and a character count, which still proves captions are flowing and how fast, and
`--echo-transcript` exists for developers running the backend by hand and is never passed by
the app.

**The diagnostics export is an allow-list, not a filter.** The one feature whose purpose is to
send a file to a stranger is the place the privacy claim is easiest to break. A filter has to
anticipate every category of secret; an allow-list only emits what somebody deliberately put
on it. `backend.log` is excluded from the export entirely. Filtering it was designed and
reviewed three times, and each round found a way the filter let something through or stripped
everything useful.

**Device names are health information.** A capture device called "Headset (R-Phonak hearing
aid)" tells the reader that the user wears a hearing aid, arriving through a field nobody
thinks of as sensitive. The diagnostics report says *whether* a device was chosen, never
which. `audio.py` logs device counts and never names. This rule has to be re-applied every
time a new capture path is added.

**Speaker voiceprints are the one exception, and it is disclosed.** Pinning a speaker writes
their name and a 512-dimensional embedding to `speakers.json`. That is the whole point of
pinning and it is the only way the feature could work, but it means a named voiceprint for
another person sits on the user's disk. `PRIVACY.md` explains this at length, including that
the embedding cannot be turned back into audio.

---

## 3. How it fits together

```
microphone or system audio
        |
   resample to 16 kHz (soxr HQ)
        |
   AudioConditioner: DC removal, 80 Hz high-pass biquad, capped gain
        |
   Silero VAD v6, 512-sample frames, state carried across frames
        |
   utterance buffer with pre-roll and end-silence endpointing
        |
   ASR worker: provisional pass (greedy) then final pass (beam 5)
        |
   speaker embedding on the finalised audio, matched against centroids
        |
   WebSocket JSON  ->  WinUI 3 app, or the browser page, or both
```

**The backend is a separate process on purpose.** It keeps a multi-gigabyte working set out of
the UI process, and a crash in inference leaves the window alive and reconnecting rather than
taking the app down mid-conversation.

**The UI knows no model ids.** The backend sends a catalogue and the UI renders whatever
arrives. This is why adding a model is an edit to `models.py` and nothing else, and why a
second client in another language needed no new protocol.

**There are three engines behind one seam.** `engine.py` defines a four-member protocol
(`settings`, `partial`, `final`, `warmup`). `asr.py` is faster-whisper on CTranslate2,
`asr_onnx.py` is ONNX Runtime GenAI for ARM and has never been run, `asr_stream.py` is a
streaming transducer on sherpa-onnx. The seam is deliberately narrow: everything about how a
model is loaded, prompted or quantised stays inside the implementation.

**Two decoding passes, one model.** Provisional text is greedy so words appear quickly; the
final pass re-decodes the completed utterance with beam search. Only one model is resident.
Both passes are Whisper, so accent robustness holds end to end. A fast non-Whisper model for
provisional text would break down on exactly the accents it was never trained on.

---

## 4. The measurements

Every number here was produced by something in the repository and can be re-run. Where a
figure was later retracted, both are shown, because a retraction is more informative than a
correction.

### Decode latency, milliseconds

Against a 1000 ms responsiveness budget. `bench/bench_latency.py`.

| Model | CUDA (Quadro RTX 8000, fp16) | CPU 4 threads (i9-14900K, int8) | CPU 16 threads |
|---|---|---|---|
| base | 55 | 730 | 405 |
| small | 250 | 1450 | 810 |
| medium | 550 | 3460 | 2260 |
| distil-large-v3 | 500 | 4370 | 3610 |
| large-v3 | 650 | 4540 | 4400 |
| stream-en (Zipformer) | 135 | 135 | 135 |
| stream-en-kroko | 120 | 120 | 120 |

The two streaming rows carry the same figure in both CPU columns deliberately, because
`asr_stream._threads()` caps the recogniser at four threads whatever the machine has, so the
sixteen-thread column is unreachable. `tests/test_stream_engine.py` asserts that coupling, so
raising the cap fails a test rather than quietly making the picker lie.

Scaling with cores is strongly sublinear. Quadrupling threads takes large-v3 from 4540 only to
4400.

### Snapdragon X Elite, native ARM64

`bench/bench_arm.py`, results in `bench/results/`.

| Model | Balanced | Best Performance | i9-14900K |
|---|---|---|---|
| tiny | 203 | 132 | 173 |
| base | 519 | 297 | 298 |
| small | 1580 | 987 | 846 |
| medium | 5032 | 3331 | n/a |

**Power mode is worth about 1.6x**, and it moves `small` across the 1000 ms line rather than
merely near it. The recorded conclusion is to design for a mid-range machine on battery in
Balanced, not a fast one plugged in. On Best Performance the Snapdragon matches an i9-14900K
at tiny and base, so native ARM is not the degraded tier; the emulated x64 path it replaces
was.

### The Hexagon NPU

whisper-small at w8a16: encoder 713.5 ms fixed per utterance, decode step 17.46 ms times 24,
so 1132 ms total. **The decoder is excellent and the encoder dominates.** The probe round-trips
about 27 MB of cross-attention cache to the host, which a real implementation would keep
on-device with IOBinding, so the encoder figure is pessimistic. Off-the-shelf float models fall
back to CPU entirely and produce identical transcripts 4 to 11 percent slower.

### Other measured facts

| Claim | Number | Source |
|---|---|---|
| Microphone placement beats model choice | close-talking 15.95% WER against 33 to 42% at 1 to 2 m, roughly 2x | AMI corpus, `ENGINEERING.md` |
| Whisper large-v3 on Indian-accented English | 7.2% WER against 20.7% for a commercial en-IN model | SVARAH |
| Turbo's pruned decoder on harder audio | +4.5% relative WER on clean, +8.4% on varied, so 2 to 4x worse where accents live | published deltas |
| float16 against int8 on Turing | int8 is about 15% slower, not faster | `ENGINEERING.md` table |
| word_timestamps cost | about 2.7% on a 9.5 s clip, inside run-to-run noise | 941 ms against 916 ms best of three |
| Speaker embeddings, same speaker under 2 s | 0.33 similarity, no better than chance | `speaker.py` docstring |
| Speaker embeddings, overlapping windows | 0.82 | same |
| Speaker embeddings, different speakers | 0.34 | same |
| Speaker labelling accuracy | about two thirds of turns on two-speaker audio | `ENGINEERING.md` |
| Prefix churn, Zipformer | rewrites text already on screen 147 times in 251 refreshes | `bench/bench_stream_churn.py` |
| Prefix churn, Kroko | 44 in the same 251 | same |
| Gain drift within one utterance | one loud word moved already-decoded samples by 67% | `asr_stream.py` docstring |
| soxr HQ against QQ resampling 44.1 to 16 kHz | 81.4 dB against 73.9 dB | `loopback.py` |
| CUDA payload trimmed | 828 MB shipped of 1,984 MB installed | PE import table analysis |
| scipy removed | 128 MB of payload for one biquad | `preprocess.py` |

### Numbers that were retracted

**Streaming churn "0 of 31".** The engine docstring claimed the transducer rewrote nothing.
That was measured on one synthesised clip and did not survive real speech, where it is 147 of
251. The lesson recorded at the time: replacing one unbacked number with another is not much
of an improvement, so `bench_stream_churn.py` now produces it.

**Kroko at 80 ms.** A first measurement took the median of three passes. The model has an
occasional fast mode and short runs land in one mode or the other, so three passes pinned
nothing. Over eight consecutive passes Kroko sits at 120 to 125 and stream-en at 132 to 144.
This mattered beyond honesty: at 80 ms Kroko was the fastest entry in the whole catalogue by a
wide margin, which is what let it win the picker's fastest-first branch outright.

**stream-en at 180 ms.** No longer reproduced on its own script and clip. Restated with Kroko
in one run, because two figures in the same column measured different ways are worse than
either being stale.

**Three claims in the `hardware.py` latency docstring.** Corrected in `7a5f426`, comments only.
It said lag barely varies with utterance length because Whisper pads to 30 s: only the encoder
does, and base measures 44, 43, 67, 99 and 155 ms for 1, 2, 5, 10 and 20 seconds. It said the
tables describe the provisional pass: they describe neither pass as shipped. It said the final
pass adds a constant: raising the beam to 5 and enabling word timestamps costs 1.4x to 3.4x and
the multiplier grows with length. **The consequence is still live: a measured figure and a
table figure are not on one scale**, which is why the dev box records base at 249 ms against a
table entry of 55 ms and then lists base as slower than small.

---

## 5. Decisions, and the reasoning behind them

### Whisper large-v3 rather than anything faster

The whole product rests on this. Turbo prunes the decoder from 32 layers to 4, which costs
little on clean speech and 2 to 4 times more on the harder, varied audio where accents live.
The English-only streaming models publish no multi-accent benchmarks at all. Accent robustness
is the reason this app exists in the form it does, so a model change that trades it is not a
tuning decision.

### No software noise suppression

Do not route audio through NVIDIA Broadcast or similar before this pipeline. Neural denoising
measurably lowers transcription accuracy and disproportionately clips accented speakers and
short utterances, which is precisely this workload. Hardware beamforming inside a microphone
array is fine. What the pipeline does apply is deliberately conservative: DC removal, an 80 Hz
high-pass to strip rumble below the speech band, and gentle level normalisation for quiet
far-field audio.

### Context as `initial_prompt`, never `condition_on_previous_text`

Recent finalised text is fed back to improve proper nouns and continuity. It is capped at 220
characters and expires after 30 seconds of silence, because unbounded feedback of the model's
own output seeds hallucination loops. That is the same reason `condition_on_previous_text`
stays off.

### The caption convention is BBC and DCMP live subtitling

Provisional text is dimmed and italic and is replaced exactly once, at the utterance boundary,
by full-contrast final text. Corrections batch at sentence boundaries rather than churning word
by word, which is what accessibility guidance recommends for deaf and hard-of-hearing readers.
Target latency is under the BBC's 1.5 s tolerance.

### It is a pause, not a stop

The backend maps the command to `controller.pause()`, which clears an Event. The model and the
ASR worker stay resident, so resuming costs a fraction of a second rather than the half minute a
cold start takes. A stop square promised otherwise, which discourages exactly what the control
is for: ducking out of a conversation for an aside and coming back. **The microphone genuinely
is released either way**, because the capture source is a context manager that closes when the
frame loop ends, so Windows' in-use indicator switches off and people in the room can see it.

### Model descriptions say what a model is, never whether to pick it

`large-v3` once ended with "Recommended." That is true on a GPU and wrong on every laptop,
where the same screen had already measured it at five seconds behind speech and filed it under
the models that cannot keep up. Which model to pick depends on the machine, the app measures
that, and a fixed word cannot know.

### The picker sorts by what this PC can keep up with

The headings deliberately do not mention processors or graphics cards. An earlier draft said
"Recommended for your processor", which asks somebody who has just installed a captioning app
to know which part of their PC does the work. Nothing is hidden: a machine that struggles today
may be plugged in tomorrow, and hiding an option from somebody who has read the delay is
deciding for them.

### `auto_select` is a wire field, not a rule in one function

The app must never choose a model whose publisher has declared no licence. That guarantee was
first put in `hardware.default_model`, which turns out to be dead code because both launchers
always pass `--model`. The screen that actually preselects is the first-run picker in the
frontend, and its fastest-first fallback ignores catalogue order, so it would have chosen Kroko
on exactly the slow machines these models exist for. The flag now rides on the catalogue
payload, next to the data it constrains.

### Speaker ids are handed out from a counter and never reused

They are deliberately not positions in a list. Merging used to delete from the middle and
renumber everybody above, while the UI relabels captions by matching on id, so folding two
speakers together silently re-attributed other people's already-scrolled lines. In a transcript,
for a deaf user, that is the only record of who said what.

**The ordering guarantee that follows:** `speaker_merged` and `speaker_deleted` are always
emitted *before* the roster that reflects them, so a client can move already-displayed lines onto
the surviving id first. Handle them the other way round and those captions keep the name of
somebody the user just asked the app to forget.

### Two window geometries, remembered separately

Compact and expanded are different windows to the person using them, one sized to work in and
one parked over a video. Sharing a geometry drags each into the other's shape at every switch.

Geometry is only recorded from a window actually showing what the user arranged. Before first
activation it describes nothing; minimised, Windows reports the window at -32000,-32000 sized
160x28, which is a real positive size and walks straight past a width and height check;
maximised, it reports the whole work area, which in compact would persist a full-screen caption
strip and permanently lose the small footprint the mode exists for.

### Three ways out of compact mode, all always available

The expand button, Ctrl+Shift+C, and the menu item. A window with no menu and no command bar is
a bad place to discover you are stuck. Entering is refused whenever the setup page, the settings
page or a dead engine owns the window, because compact would hide the thing that has to be dealt
with first. Being forced out that way does not count as changing your mind, so the preference
survives it.

---

## 6. Reversals: decisions that were made twice

This section is the most useful thing in this document, because each entry is a place where the
obvious answer was wrong and the reasoning is not reconstructable from the current code.

### Microphone consent: the app asks, then Windows asks

**First position** (`aa6e385`): Windows will not raise a consent dialog for a `runFullTrust`
packaged app. Probing a live package across never-asked, Prompt and Deny returned Allowed every
time without prompting, and the Camera-style prompt was believed to be AppContainer behaviour.
So the app asked with a dialog of its own.

**Reversal** (`5ddad6c`): that is wrong. Windows raises its own prompt the first time the device
is actually opened. Observed on a packaged install, the per-app consent entry went from Prompt to
Allow at the moment capture started, with no involvement from the app. So there were two dialogs
for one decision and only the uncontrolled one was recorded anywhere.

**Current position:** the app never asks. An undecided state means "go ahead and try", the
microphone is opened, and Windows asks natively in the place people recognise. What remains is
reporting a refusal Windows has already stored, because those never re-prompt and the only remedy
is the Settings app.

**The related trap:** `CheckAccess()` is documented never to prompt and returns
`UserPromptRequired` when there is no answer on file. Treating that as a refusal is how the app
used to send people to Settings for a permission they had never been offered.

### The stall warning: exempted, then made precise

**First position** (`55e4a69`): WASAPI delivers no frames from an output endpoint while nothing
is playing, so on loopback a quiet desktop is indistinguishable from a dead capture thread. The
"No audio" warning fired every time playback stopped. Fix: exempt loopback from the stall check.

**Reversal** (`32fc5a5`): that removed the false alarm and the true one together. When the Phonak
headphones dropped off Bluetooth mid-session, the app sat showing a running clock and "Listening
for speech" above a transcript that could never gain another line.

**Current position:** distinguish an idle endpoint from a dead one. `LoopbackStream.is_alive`
checks whether the stream is still active, and the frame loop yields real silence while idle so
level reporting stays alive. Silence is the truthful description of an idle output.

### The model picker: inline, flyout, inline again

Started inline. The expander whipped, because a WinUI `Expander` in an `Auto` row above a
star-sized list makes the row snap to its final size while the expander animates its content.
Moved to a flyout to escape it. Moved back inline with one eased height animation driving a
clipping `ScrollViewer`, so the star row follows frame by frame.

**The detail that matters:** layout animations need `EnableDependentAnimation` or they are
silently dropped, which is exactly the jump this replaced.

### The wandering tooltip: blamed twice on the wrong thing

A tooltip reading "Ctrl++" appeared over the middle of the window and followed the pointer. Two
commits blamed the overflow menu's `KeyboardAcceleratorTextOverride` and neither removed it,
because it was never coming from the menu. WinUI shows a keyboard accelerator's combination as a
tooltip on whichever element owns it, and Sunno's caption-size accelerators are owned by the
content root so they work wherever focus is. Setting the placement mode to Hidden fixed it.

Separately, `KeyboardAcceleratorTextOverride` really does generate a mangled tooltip: it splits
on spaces and rejoins with "+", so "Ctrl +" arrives as "Ctrl++", and the generated tooltip belongs
to the accelerator text block inside the item so an explicit tooltip on the item does not suppress
it. The shortcut is now part of the label text, so no such element exists.

### The hallucination filter: three rounds, each deleting real speech

Whisper invents caption credits over near-silence because it was trained on subtitled video.
Observed live: "Subtitling by SUBS Hamburg", "Subtitles by the Amara.org community".

**Round one:** exact-match blocklist. Could never work, because the wording varies.

**Round two:** substring matching anywhere in a segment. Deleted "The book was translated by
Tolkien himself", with no trace, from somebody who cannot hear what was said.

**Round three:** anchored to the segment opening plus an eight-word cap. Still deleted "Subtitles
by default are off" and "Subscribe to that podcast, it's really good".

**Current position:** the discriminator is what the attribution looks like. A caption credit names
a studio or a handle; ordinary speech attributes to common words (by default, by hand, by my
sister). So a credit opener also requires the following word to be capitalised, which is why the
original text is inspected rather than a lower-cased copy. Matching is per sentence rather than
per segment, so a credit appended to real speech is caught while a mid-sentence mention is not.
Tokens that are never speech (amara.org, castingwords, bare URLs) still match anywhere.

**The test lesson:** the original test gave false confidence because its pass list contained no
"<word> by" construction at all. It now carries 16 credits blocked and 23 real sentences kept,
including every one the reviewer demonstrated was being deleted.

### Model switch: committed too early, twice

**First position:** persist the new model to settings when it is chosen. A model that downloaded
but failed to load became the choice reloaded on every launch, an unrecoverable crash loop for
somebody who cannot be expected to hand-edit `settings.json`.

**Second position:** commit on socket connect. Still too early. The backend accepts WebSocket
connections before it loads the engine, roughly half a minute before the model is usable, so any
load failure slower than the handshake still recorded a broken model.

**Current position:** a switch completes on the first status reporting `listening` or `stopped`,
which is the only signal meaning the engine actually loaded.

**The related fix:** the fallback no longer targets a hardcoded large-v3. That model can
legitimately be absent, since first run lets you pick any single model and large-v3 is 3 GB, so
falling back to it would swap a crash for a 3 GB download prompt and persisting it would make
every future launch open that prompt instead of captioning. It now prefers the last model seen to
load, then any model on disk.

### The memory leak fix that was withdrawn

`WordInlines` kept a plain dictionary of `CaptionLine` to `RichTextBlock` and never pruned it,
leaking an entry and its visual subtree per finalised utterance over a session meant to run for
hours. The fix added a `ConditionalWeakTable` plus an `Unloaded` hook.

**The hook was removed in the next commit.** `Unloaded` can fire for a block still showing a
current line, and detaching there unsubscribes `PropertyChanged` with nothing to re-attach it,
silently freezing that line so a provisional caption never upgrades to its final text. The
weak-keyed table already bounds growth without that risk. Recorded reasoning: trading a certain
correctness risk for a leak that is already fixed was a bad bargain.

### Kroko: withdrawn, then shipped with disclosure

**First position** (`a72e4cb`): withdrawn. The licence does not hold up. The mirror serving the
files declares none, its readme points at Banafo, Banafo declares `license_name: test` with a link
to a LICENSE file that is zero bytes. That is not ShareAlike to be weighed, it is no grant at all,
and it is the same condition a punctuation model had already been refused over.

**Reversal** (`d05841d`): shipped anyway, as a deliberate and recorded decision by the project
owner, who was shown all of the above. What that costs is not hidden: `THIRD-PARTY-NOTICES.md`
carries a section stating the position plainly, and the app never chooses it for anybody. Banafo
also publishes only data files, so the ONNX files come from somebody else's unattributed
conversion, which even a clear grant from Banafo would not cover.

What it buys is readability rather than speed: it writes its own capitals and punctuation and
revises far less text already on screen, 44 refreshes in 251 against 147. On latency the two are
near tied.

### Caption size: rebuild, then bind

`CaptionSize` was a static non-notifying property bound one-way, which the XAML compiler flagged.
`SetFontSize` worked around it by clearing and re-adding every line, which re-ran the per-word
machinery for every line, discarded any active text selection mid-read, and scaled with transcript
length. It is now an instance property with change notification bound once on the items control and
inherited by each caption, so one notification resizes every line in place.

---

## 7. Bugs worth knowing about

Each of these cost real time to find and could recur.

**The packaged app had never worked, and the crash was invisible.** The Windows App SDK ships its
own onnxruntime at the package root, and a packaged process searches that directory, so
sherpa-onnx bound to App SDK 1.23 instead of the 1.27 sitting beside it and died with an access
violation as soon as speaker labelling initialised. **The staged tree never reproduced it, because
only an installed package searches the package root.** Both DLLs are now pruned at publish time and
the build fails if they return.

The crash was invisible because backend stdout went into a `List<string>` nothing read, there was
no `Exited` handler, and reconnect attempts kept repainting "Starting the speech engine" over the
failure. A user relying on this would wait forever.

**A second `AppCapability.CheckAccess()` call took the process down.** The first call during
construction succeeds; a repeat from the content `Loaded` handler crashes with a stowed exception
in `Microsoft.UI.Xaml` that never reaches `UnhandledException`. Packaged only. Consent is now read
once at startup and reused, which is also the better design.

**A guard that could never fire.** `register_cuda_dlls(required=True)` hit `if _registered: return`
before evaluating `required`, because the module-level call at import had already set the flag. The
guard was dead on exactly the path it was written for. Verified against the real production
sequence rather than a fresh module import.

**Every string contains the empty string.** The default-output lookup fell back to `""` and
compared with `in`, so a machine that could not determine its default marked every output device as
the default. The same two-way substring test also matched "Realtek" against "Speakers (Realtek
Audio)". Now an exact comparison on names both put through the same suffix stripping.

**`ScrollToEnd` read the extent before layout.** Every caller runs from the handler that has just
added or replaced a line, so it scrolled to the extent the list had before those words existed and
a caption wrapping to two or three rows left its tail hidden.

**One microphone appeared three times.** Devices are published once per host API with mangled
names. The fix narrows to the WASAPI enumeration, which is the only host API whose list tracks
whether the hardware is actually present: measured on one desktop, 22 input entries across four
APIs of which WASAPI saw 4, and the other 18 included two unplugged jacks and four cameras last
connected months ago.

**PortAudio fixes its device list at `Pa_Initialize`.** The process cannot see anything plugged in
since it started, and cannot simply look again, because re-initialising while a capture stream is
open invalidates that stream. The refresh runs in a child process which starts its own PortAudio,
reads the hardware and exits.

**An em-dash broke the build.** `build-msix.ps1` has one in a `Write-Host` line. Windows PowerShell
5.1 reads the file in the ANSI codepage rather than UTF-8, which corrupts the parse and fails with
a missing string terminator forty lines later. It builds under PowerShell 7.

**A test that could not fail.** The streaming prefix-stability test compared lower-cased prefixes,
which is true by construction, so it would have passed with the function deleted. It now runs a
transform that revises an earlier word once a later one arrives and asserts both that the check
catches it and that a causal version passes, which pins the failure to the lookahead. That second
assertion caught a defect in the first draft of the control immediately.

**The screen reader was never told anything.** `AutomationProperties.LiveSetting` is metadata: it
tells an assistive client how urgently to treat a change, but nothing in WinUI watches the items and
raises `LiveRegionChanged`. Captions now go through a dedicated one-line element that raises it by
hand, finals only, because speaking each provisional revision would be the audible form of words
appearing and disappearing that do not match what was said.

That element is sized 1x1 with transparent ink rather than zero-height, because a zero-area element
reports `IsOffscreen=true` and assistive clients skip offscreen providers. Verified with a UIA
client, which saw the element and received no events from it.

**The announcement used the wrong label.** It read `SpeakerLabel` while the visible line and the
clipboard both use `DisplayLabel`, so a user's own line was read out under their name while the
screen showed "You".

---

## 8. Working conventions

**No em-dashes in published prose.** Markdown files and any string the app shows a user. Code
comments are exempt, since they are read by people working on the code. This is a standing
instruction from the owner and there is a build hazard behind it (see above).

**Commit messages are prose paragraphs explaining why, not bullet lists of what.** Read `git log`
for the register. They are long, they name the alternative that was rejected, and they record what
was verified and how.

**Tests are directly runnable scripts in `tests/`, not pytest.** Exit code 0 means pass. They print
what they checked, so a suite that silently shrank on a fresh clone is visible.

**Every change goes through a review pass before commit.** The recorded experience is that this has
caught a shipping-severity bug in most rounds, including one that would have failed on exactly the
ARM model tier. Budget for two or three rounds.

**Packaging uses explicit include-lists, never directory copies.** `stage-backend.ps1` fails the
build if a dev-only package appears in the payload, because scipy, PIL and pip once leaked 155 MB.
The models directory holds about 130 MB of benchmark models that must not silently inflate the
package, so a single file is named rather than a folder copied.

**Documentation records rejected alternatives.** This is why `ENGINEERING.md` explains why
large-v3 rather than turbo, and why `models.py` carries a paragraph about a punctuation model that
is not used.

**Benchmarks live in the repository.** They previously lived only on the author's machine, so the
comment telling the next person to re-run them pointed at nothing.

**Build proof venvs from `requirements.txt`, never by hand.** A hand-built venv is structurally
incapable of catching a broken requirements file, and that is exactly how a deleted
`onnxruntime>=1.17` got through a review round.

---

## 9. Landmines

**`build-msix.ps1:33`'s `\x64\` filter must not be templated.** It selects the host
`makeappx`/`signtool` on the dev box. Changing it breaks the build.

**Unknown model ids fall through to `_UNKNOWN_MODEL_LAG_MS = 5000`**, so the picker shows every
model as "5 s, not responsive". The ARM port hit exactly this.

**`record_latency` and the shipped tables are on different scales.** See section 4. Do not
reconcile them by remeasuring one side and leaving the other.

**`cuda_setup.py:79` raises a RuntimeWarning about missing NVIDIA libraries on every import in a
non-CUDA environment.** Harmless, but it is the first thing a non-NVIDIA user sees and it is
misleading there.

**The QNN incantation is three steps and all are load-bearing.** `providers=["QNNExecutionProvider"]`
only reaches EPs compiled into the build and fails with "not compatible with any execution provider
added to the session" even when the name appears in `get_available_providers()`. Register the
library, select by device, and pass `backend_path`. See `ARM-PORT.md`.

**genai will not load onnx-community Whisper exports.** They are Optimum or Transformers.js graphs
with no `genai_config.json`. Use `bench/convert_whisper_genai.py`. That converter needs two
undocumented workarounds, both already handled: transformers 5.x asks the Hub with `token=True` even
for public weights, and the builder deletes its cache directory when empty while the Whisper path
saves twice.

**Converting a genai Whisper needs torch, which has no `win_arm64` wheel.** Conversion is an x64
build-time step, so where ARM weights come from is an open question rather than an assumption.

---

## 10. Where the project stands

**Windows x64:** shipping, Microsoft Store, version 1.0.76 at the time of writing.

**Windows on ARM:** unfinished. `docs/ARM-PORT.md` has six open items, including an `OnnxEngine`
that is written and has never been run, a model catalogue that is not architecture-aware, three
native dependencies with no ARM path, a staging script that cannot build an ARM tree, and an ARM
refusal message that is still shipping and will be wrong the moment an ARM build exists.

**Known product limitations**, stated in the README rather than hidden: speaker labels are
best-effort at about two thirds of turns; overlapping speech degrades badly for single-channel
recognition of any kind; utterances longer than `max_utterance_s` split mid-sentence; and
provisional lines carry no speaker chip because identification happens at commit time.

**The largest outstanding architectural debt** is in the streaming path. `asr_stream.py` rebuilds a
fresh recogniser over the whole utterance on every partial, because the pipeline hands out whole
utterances and the conditioner renormalises gain over whatever it is given. Doing it properly means
the pipeline handing out deltas with a committed prefix, which is a change to a seam all three
engines share. The file's own docstring describes the fix. This is worth more than any model swap in
that tier, and it is the reason the churn numbers look the way they do.
