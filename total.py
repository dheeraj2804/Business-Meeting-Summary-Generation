import whisper
import ssl
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from heapq import nlargest
import torch
from simplet5 import SimpleT5
from transformers import T5ForConditionalGeneration, T5Tokenizer

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

