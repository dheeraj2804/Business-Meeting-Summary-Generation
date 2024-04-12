import whisper
import spacy
import ssl
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest
import torch
from simplet5 import SimpleT5
from transformers import T5ForConditionalGeneration, T5Tokenizer
from pyannote.audio import Pipeline
from pydub import AudioSegment
import re
import webvtt
import subprocess


ssl._create_default_https_context = ssl._create_unverified_context

def process_audio(filepath):
    model = whisper.load_model("base")
    result = model.transcribe(filepath)
    res = result["text"]
    return res

def extractive_summarization(transcript):
    stopwords = list(STOP_WORDS)
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(transcript)

    tokens = [token.text for token in doc]

    word_freq = {}
    for word in doc:
        if word.text.lower() not in stopwords and word.text.lower() not in punctuation:
            if word.text not in word_freq.keys():
                word_freq[word.text] = 1
            else:
                word_freq[word.text] += 1

    max_freq = max(word_freq.values())

    for word in word_freq.keys():
        word_freq[word] = word_freq[word]/max_freq

    sent_tokens = [sent for sent in doc.sents]

    sent_scores = {}
    for sent in sent_tokens:
        for word in sent:
            if word.text in word_freq.keys():
                if sent not in sent_scores.keys():
                    sent_scores[sent] = word_freq[word.text]
                else:
                    sent_scores[sent] += word_freq[word.text]

    select_len = int(len(sent_tokens) * 0.3)
    summary = nlargest(select_len, sent_scores, key=sent_scores.get)
    final_summary = [word.text for word in summary]
    summary = ' '.join(final_summary)

    return summary

def abstractive_summarization(text, max_chunk_length=1500, max_summary_length=300):
    model_name = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    chunks = [text[i:i+max_chunk_length] for i in range(0, len(text), max_chunk_length)]
    summaries = []
    for chunk in chunks:
        input_ids = tokenizer.encode("summarize: " + chunk, return_tensors="pt", max_length=512, truncation=True)
        output = model.generate(input_ids, max_length=max_summary_length, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(output[0], skip_special_tokens=True)
        summaries.append(summary)

    final_summary = " ".join(summaries)

    return final_summary

def abstract(text):
    text = "summarize: " + text
    model = SimpleT5()
    model.from_pretrained(model_type="t5", model_name="t5-base")
    model.load_model("t5","outputs/simplet5-epoch-2-train-loss-0.9274-val-loss-1.4256", use_gpu=False)
    res = model.predict(text)
    result = res[0]
    
    return result

# def perform_speaker_diarization(audio_file_path, audio, output_audio_path='dz.wav', output_vtt_path='dz.vtt'):
#     # Load the pre-trained speaker diarization pipeline
#     pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
#                                          use_auth_token="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")
    
#     # Perform speaker diarization
#     dz = pipeline({'uri': 'blabal', 'audio': audio_file_path})
    
#     # Write diarization results to a text file
#     with open("diarization.txt", "w") as text_file:
#         text_file.write(str(dz))
    
#     # Function to convert time string to milliseconds
#     def millisec(timeStr):
#         spl = timeStr.split(":")
#         s = (int)((int(spl[0]) * 60 * 60 + int(spl[1]) * 60 + float(spl[2])) * 1000)
#         return s
    
#     # Process diarization results and generate audio segments
#     spacer = AudioSegment.silent(duration=100)  # adjust the duration as needed
#     sounds = spacer
#     segments = []
    
#     dz = open('diarization.txt').read().splitlines()
#     for l in dz:
#         start, end = tuple(re.findall(r'[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=l))
#         start = int(millisec(start))  # milliseconds
#         end = int(millisec(end))  # milliseconds

#         segments.append(len(sounds))
#         sounds = sounds.append(audio[start:end], crossfade=0)
#         sounds = sounds.append(spacer, crossfade=0)
    
#     # Export segmented audio to a WAV file
#     sounds.export(output_audio_path, format="wav")
    
#     # Use the generated audio file for captioning
#     subprocess.run(['whisper', output_audio_path, '--language', 'en', '--model', 'base'])
    
#     # Read captions from the generated VTT file
#     captions = [[(int)(millisec(caption.start)), (int)(millisec(caption.end)), caption.text] for caption in webvtt.read('dz.vtt')]
    
#     # Return the captions
#     return captions 

def perform_speaker_diarization(audio_file_path, audio, output_audio_path='dz.wav', output_vtt_path='dz.vtt'):
    # Load the pre-trained speaker diarization pipeline
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                         use_auth_token="Authenticated Token")
    
    # Perform speaker diarization
    dz = pipeline({'uri': 'blabal', 'audio': audio_file_path})
    
    # Write diarization results to a text file
    with open("diarization.txt", "w") as text_file:
        text_file.write(str(dz))
    
    # Function to convert time string to milliseconds
    def millisec(timeStr):
        spl = timeStr.split(":")
        s = (int)((int(spl[0]) * 60 * 60 + int(spl[1]) * 60 + float(spl[2])) * 1000)
        return s
    
    # Process diarization results and generate audio segments
    spacer = AudioSegment.silent(duration=100)  # adjust the duration as needed
    sounds = spacer
    segments = []
    
    dz = open('diarization.txt').read().splitlines()
    for l in dz:
        start, end = tuple(re.findall(r'[0-9]+:[0-9]+:[0-9]+\.[0-9]+', string=l))
        start = int(millisec(start))  # milliseconds
        end = int(millisec(end))  # milliseconds

        segments.append(len(sounds))
        sounds = sounds.append(audio[start:end], crossfade=0)
        sounds = sounds.append(spacer, crossfade=0)
    
    # Export segmented audio to a WAV file
    sounds.export(output_audio_path, format="wav")
    
    # Use the generated audio file for captioning
    subprocess.run(['whisper', output_audio_path, '--language', 'en', '--model', 'base'])
    
    # Read captions from the generated VTT file
    captions = [[(int)(millisec(caption.start)), (int)(millisec(caption.end)), caption.text] for caption in webvtt.read('dz.vtt')]
    
    # Add speaker labels to captions
    speaker_labels = []
    spacermilli = 100  # Define spacermilli value (adjust as needed)
    for i in range(len(segments)):
        idx = 0
        for idx in range(len(captions)):
            if captions[idx][0] >= (segments[i] - spacermilli):
                break;
        
        while (idx < (len(captions))) and ((i == len(segments) - 1) or (captions[idx][1] < segments[i+1])):
            c = captions[idx]  
            
            start = segments[i] + (c[0] - segments[i])

            if start < 0: 
                start = 0
            idx += 1

            start = start / 1000.0
            startStr = '{0:02d}:{1:02d}:{2:02.2f}'.format((int)(start // 3600), 
                                                    (int)(start % 3600 // 60), 
                                                    start % 60)
            speaker_label = "[Speaker1]" if i % 2 == 0 else "[Speaker2]"
            speaker_labels.append(f"{startStr} {speaker_label} {c[2]}")
    
    # Return the captions with speaker labels
    return speaker_labels











