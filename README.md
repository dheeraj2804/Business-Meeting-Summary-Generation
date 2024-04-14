
# Project Name: Business Meeting Summary Generation.


## Description

This project involves the development of an innovative system for meeting summary generation. The system couples the audio-to-text transcription with both extractive and abstractive summarization pipelines to generate different kinds of summaries for easy understanding and interpretation. The system is also able to generate a meeting agenda that helps in better understanding the context of the meeting.
## Software Requirements

1. Editor for HTML, CSS, Python, Flask - VS Code
2. Google Chrome, Firefox, Microsoft Edge or Brave Browser
3. Google collab and kaggle notebooks for training.
4. Python(Version 3.0 or Greater)
5. Flask Library
6. Keras/TensorFlow Library
7. Pandas
8. Torch Library
9. NLTK Library
10. Whisper Library
11. pyannote.audio Library
## Installations

* Clone or download the project repository from GitHub.
* Install the required software dependencies listed in the "Software Requirements" section.

Numpy:
```bash
pip install numpy
```
Pandas:
```bash
pip install pandas
```
nltk:
```bash
pip install nltk
```
```bash
pip install scikit-learn
```
```bash
pip install spacy
```
Sentencepiece:
```bash
pip install sentencepiece
```
Pydub:
```bash
pip install pydub
```
-OR-
```bash
pip install git+https://github.com/jiaaro/pydub.git@master
```
-OR-
```bash
git clone https://github.com/jiaaro/pydub.git
```
Transformers
```bash
pip install transformers
```
SimpleT5
```bash
pip install --upgrade simplet5
```
Whisper
```bash
pip install -U openai-whisper
```
-OR-

```bash
pip install git+https://github.com/openai/whisper.git
```
Pyannote
```bash
pip install pyannote.audio
```

* Create your access token on huggingface to utilize their pyannote-Speaker Diarization system by following their terms and conditions. (Follow these instructions [Speaker-Diarization](https://huggingface.co/pyannote/speaker-diarization-3.1) for more understanding)
* Set up the development environment on your development machine.
## Usage

1. Navigate to the project directory on your development machine.
2. To start the **Main Page of Project** in the live server:

    * Run the *main.py* script using the command ***python main.py***
      ```bash
      python app.py
      ```
    * This will start the server and you can access the website page through the provided URL at localhost:5000.
3. We have divided our project into 3 main services comprising:
   
    * Audio to Text Transcription service.

    * Extractive Summarization service.

    * Meeting Agenda Generation(Abstractive summarization) service.
4. To access the first service i.e Audio-to-text Transcription

    * Navigate through the website to access the 1st service page and upload the audio file and the system generates the text transcript with the speaker labels along with the time stamps of different speakers speaking.

5. To access the second service i.e Extractive Summarization

    * Navigate through the website to access the 2nd service page and upload the audio file and the system generates an extractive summary.

6. To access the third service i.e Meeting Agenda Generation

    * Navigate through the website to access the 3rd service page and upload the audio file and the system generates a Meeting agenda along with a short summary of abstractive type.

7. To close the server press ***ctrl+c*** in the terminal.

## Execution Screenshots
1. Main page
![Main_Page.jpg](https://github.com/dheeraj2804/Business-Meeting-Summary-Generation/blob/master/Project%20Execution%20Screenshots/UI1.png)

2. Services Page

![Services_Page.jpg](https://github.com/dheeraj2804/Business-Meeting-Summary-Generation/blob/master/Project%20Execution%20Screenshots/UI2.png)

3. About Page

![About_Page.jpg](https://github.com/dheeraj2804/Business-Meeting-Summary-Generation/blob/master/Project%20Execution%20Screenshots/UI3.png)


4. Audio-to-Text Transcription service page

![Service-1_Page.jpg](https://github.com/dheeraj2804/Business-Meeting-Summary-Generation/blob/master/Project%20Execution%20Screenshots/service1-pic.png)

5. Extractive Summarization service page

![Service-2_Page.jpg](https://github.com/dheeraj2804/Business-Meeting-Summary-Generation/blob/master/Project%20Execution%20Screenshots/service2-pic.png)

6. Meeting Agenda Generation service page

![Service-3_Page.jpg](https://github.com/dheeraj2804/Business-Meeting-Summary-Generation/blob/master/Project%20Execution%20Screenshots/service3-pic.png)



*To see all the execution photos open [Project Execution Photos](https://github.com/dheeraj2804/Business-Meeting-Summary-Generation/tree/master/Project%20Execution%20Screenshots) directory*

## Credits

* A. Dheeraj Reddy [GitHub](https://github.com/dheeraj2804)
* A. Chinmaye [GitHub](https://github.com/Chinmaye09)
* D. Sakth Reddy [GitHub](https://github.com/saketh-dr)
* Mohammed Irfan [GitHub](https://github.com/irfanmd17)
