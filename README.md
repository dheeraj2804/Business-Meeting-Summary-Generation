# Business-Meeting-Summary-Generation
MLRIT Final Year Project 2023-24, Team reference no. CSM07 .  Project By: A Dheeraj, A Chinmaye, D Saketh, Mohammad Irfan

Speaker diarization 3.1
This pipeline is the same as pyannote/speaker-diarization-3.0 except it removes the problematic use of onnxruntime.
Both speaker segmentation and embedding now run in pure PyTorch. This should ease deployment and possibly speed up inference.
It requires pyannote.audio version 3.1 or higher.

It ingests mono audio sampled at 16kHz and outputs speaker diarization as an Annotation instance:

stereo or multi-channel audio files are automatically downmixed to mono by averaging the channels.
audio files sampled at a different rate are resampled to 16kHz automatically upon loading.
Requirements
Install pyannote.audio 3.1 with pip install pyannote.audio
Accept pyannote/segmentation-3.0 user conditions
Accept pyannote/speaker-diarization-3.1 user conditions
Create access token at hf.co/settings/tokens.
Usage
# instantiate the pipeline
from pyannote.audio import Pipeline
pipeline = Pipeline.from_pretrained(
  "pyannote/speaker-diarization-3.1",
  use_auth_token="HUGGINGFACE_ACCESS_TOKEN_GOES_HERE")

# run the pipeline on an audio file
diarization = pipeline("audio.wav")

# dump the diarization output to disk using RTTM format
with open("audio.rttm", "w") as rttm:
    diarization.write_rttm(rttm)

Processing on GPU
pyannote.audio pipelines run on CPU by default. You can send them to GPU with the following lines:

import torch
pipeline.to(torch.device("cuda"))

Processing from memory
Pre-loading audio files in memory may result in faster processing:

waveform, sample_rate = torchaudio.load("audio.wav")
diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate})

Monitoring progress
Hooks are available to monitor the progress of the pipeline:

from pyannote.audio.pipelines.utils.hook import ProgressHook
with ProgressHook() as hook:
    diarization = pipeline("audio.wav", hook=hook)

Controlling the number of speakers
In case the number of speakers is known in advance, one can use the num_speakers option:

diarization = pipeline("audio.wav", num_speakers=2)

One can also provide lower and/or upper bounds on the number of speakers using min_speakers and max_speakers options:

diarization = pipeline("audio.wav", min_speakers=2, max_speakers=5)

