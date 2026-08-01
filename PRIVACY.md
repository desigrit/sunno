# Privacy Policy for Sunno

**Last updated: 1 August 2026**

Sunno is a live captioning app for Windows. It listens to a microphone, or to your PC's own
audio, and writes down what it hears so you can read it.

This policy describes what Sunno does with that information. The short version is that it stays
on your computer, and this document exists to say exactly how.

---

## Who this policy is for

Two groups of people, not one.

The first is you, the person who installed Sunno. The second is everyone whose voice it captures:
the people at your table, in your meeting, or on the other end of a call. They did not install
anything and did not agree to anything, so the second group is the one this policy is really
written for.

---

## What Sunno collects

**Audio from the input you choose.** Either a microphone or, if you select it, the sound your PC
is playing. Audio is held in memory only for as long as it takes to recognise it, in chunks of a
few seconds. It is never written to disk and never sent anywhere.

**The text of what was said.** Recognition happens on your computer, using a speech model stored
on your computer. The resulting captions are shown in the app window and held in memory for the
current session.

**Voice characteristics, to tell speakers apart.** To label who is speaking, Sunno computes a
numerical fingerprint of each voice, called an embedding. For most speakers this exists only in
memory and is gone when Sunno closes.

There is one exception, and it is worth reading. If you **pin a speaker**, giving them a lasting
name so Sunno recognises them in future sessions, that person's name and voice fingerprint are
saved to `%LOCALAPPDATA%\Sunno\speakers.json` and loaded again next time. That is the whole point
of pinning, and it is the only way the feature could work, but it does mean a named voiceprint
for that person is stored on your computer until you delete it.

The fingerprint is a list of numbers describing vocal characteristics. It cannot be turned back
into audio and cannot be used to reconstruct anything that was said. It can, however, be used to
recognise that person's voice again, which is exactly what it is for.

Speakers you have not pinned are never written to disk.

**Names you type.** A name you give a speaker is kept for the session so the transcript stays
readable. It is written to disk only for pinned speakers, as described above.

---

## What Sunno sends over the network

**Almost nothing, and never your audio or your captions.**

There is exactly one network operation in the entire app: downloading the speech recognition
model the first time you use it, or when you choose a different one. That download comes from
Hugging Face, a public model host. It transfers model files to you. It does not send them
anything about you beyond what any file download requires, and it is the only time Sunno uses
the internet.

Once the model is on your computer, you can disconnect from the internet permanently and Sunno
will work exactly as before.

There is no account, no sign-in, no analytics, no telemetry, no crash reporting service, no
advertising, and no server operated by us. There is nowhere for your conversations to go, because
no such destination exists in the software.

---

## What Sunno stores on your computer

In `%LOCALAPPDATA%\Sunno`:

| File | Contents |
|---|---|
| `settings.json` | Your preferences: chosen model, input device, caption text size, always-on-top |
| `hardware.json` | Timing measurements of your PC, used to estimate caption delay |
| `speakers.json` | Pinned speakers only: name, voice fingerprint, and whether that person is you |
| `backend.log` | Diagnostic output, so a crash can be investigated |
| `backend.log.1` | The previous diagnostic log, kept when the current one is rotated |
| `startup-trace.log` | A few lines recording how far startup got, for diagnosing launch failures |

**On the input device in `settings.json`.** Sunno records the *name* of the microphone you chose,
not only its position in the list, because Windows renumbers audio devices whenever the set of
them changes and the app would otherwise silently start listening to a different one. Device
names sometimes describe the hardware in ways you might not expect, for example
"Headset (R-Phonak hearing aid)". That name stays on your computer and is never sent anywhere.
It is also deliberately left out of the diagnostics report described below.

`speakers.json` does not exist until you pin someone. Delete the file to forget every pinned
speaker at once.

The speech model itself is cached separately, in the Hugging Face cache directory under your user
profile. It is model data and contains nothing about you.

**On `backend.log`, specifically.** This file records that captions were produced, how quickly,
how confident the recogniser was, and how many characters long each one was. It deliberately does
**not** record the words themselves or the names of speakers. Developers running the backend
directly can pass `--echo-transcript` to print recognised text to their own console, but the app
never does this and never writes transcript text to disk.

You can delete any of these files at any time. Sunno will recreate what it needs.

---

## The diagnostics report

Sunno can produce a short report to attach to a bug report, from the overflow menu, "Copy
diagnostics". It is shown to you in full before it is copied, so you can read exactly what you
would be sharing.

It contains the app version and package identity, your Windows version, processor architecture and
count, the .NET version, whether the GPU or the processor is being used, which speech model is
loaded and which is selected, whether the engine is running and connected, whether the source is a
microphone or system audio, whether a device is chosen and whether its name is stored, whether the
microphone consent question has been asked, your caption size and always-on-top preference,
whether a vocabulary is set, and the timing measurements from `hardware.json`.

It deliberately contains **none** of the following: transcript text, speaker names, voice
fingerprints, vocabulary entries, device names, or any part of the log files. Where a device is
concerned it reports only *whether* one is chosen, never which one.

**Transcripts are not saved.** When you close Sunno, or clear the transcript, the captions are
gone. If you want to keep something, select it and copy it out yourself.

---

## Who your information is shared with

Nobody. It is not sold, rented, traded, or disclosed, because it never leaves your computer for
us to do any of that with.

---

## Your controls

- **Stop it listening at any time** with the pause button in the app. This releases the
  microphone rather than just hiding captions.
- **Revoke microphone access entirely** in Windows Settings, Privacy and security, Microphone.
  Sunno has its own entry there, separate from other desktop apps, so you can see when it is
  using the microphone and cut it off in one click.
- **Clear the transcript** from the app's overflow menu.
- **Forget a pinned speaker** by deleting `%LOCALAPPDATA%\Sunno\speakers.json`, which removes
  every stored name and voice fingerprint.
- **Delete stored data** by removing `%LOCALAPPDATA%\Sunno`.
- **Uninstall** through Windows Settings, Apps. Nothing is left behind on any server, because
  nothing was ever put on one.

---

## Recording other people

Sunno is a tool for understanding conversations, and it is designed so you can be honest with the
people in the room: nothing is uploaded and nothing is kept.

Whether you may capture a conversation at all is a matter of local law, and laws on recording and
consent differ considerably between countries and between states. It is your responsibility to
know and follow the rules where you are. Telling people that you are using captions is generally
both the decent and the safe thing to do.

---

## Children

Sunno is not directed at children and does not knowingly collect anything from them. It does not
knowingly collect anything from anyone, which is rather the point.

---

## Changes to this policy

If Sunno's behaviour changes in a way that affects this document, this document changes with it,
and the date at the top is updated. Because the source code is public, you can also read the
history of both.

---

## Verifying any of this

You do not have to take my word for it. Sunno is open source under the MIT licence, so you can
read exactly what it does with the microphone. If you find something that contradicts this
policy, that is a bug and I want to know about it.

---

## Contact

Open an issue at https://github.com/desigrit/sunno/issues
