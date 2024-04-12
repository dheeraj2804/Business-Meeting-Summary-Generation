from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import total
from pydub import AudioSegment
from total import process_audio
from total import extractive_summarization
from total import abstractive_summarization
from total import abstract
from total import perform_speaker_diarization



app = Flask(__name__, template_folder="templates")


app.config['UPLOAD_FOLDER'] = '/Users/dheerajreddy/Desktop/sample-copy/audio_folder'
ALLOWED_EXTENSIONS = {'mp3', 'wav'}

# Function to check if a file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Define a route for the main page

@app.route('/')
def index():
    return render_template('index.html')

# Define a route to handle file upload
@app.route('/aud2text', methods=['POST', 'GET'])
def aud2text():
    if request.method == 'POST':
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        
        if file and allowed_file(file.filename):
            # Save the uploaded file
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
            file.save(file_path)
            
            # Load the audio data
            audio = AudioSegment.from_file(file_path)
            
            # Perform speaker diarization
            result = total.perform_speaker_diarization(file_path, audio)
            
            # Remove the uploaded file
            os.remove(file_path)
            return render_template('aud2text.html', result=result)
        else:
            return 'Invalid file format'
    
    return render_template('aud2text.html')
    


@app.route('/aud2sum', methods=['POST','GET'])

def aud2sum():
    if request.method == 'POST':
        file = request.files['file']

    # If user does not select file, browser also submits an empty part without filename
        if file.filename == '':
            return 'No selected file'
        
        if file and allowed_file(file.filename):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
            file.save(file_path)
            # Process the uploaded audio file to text
            audio_text = total.process_audio(file_path)
            # Generate abstractive summarization of the text
            summary = total.extractive_summarization(audio_text)
            os.remove(file_path)
            return render_template('aud2sum.html', summary=summary)
        else:
            return 'Invalid file format'
        
    return render_template('aud2sum.html')

@app.route('/headline', methods=['POST','GET'])

def headline():
    # Check if the POST request has the file part
    if request.method == 'POST':
        file = request.files['file']

        # If user does not select file, browser also submits an empty part without filename
        if file.filename == '':
            return 'No selected file'
        
        if file and allowed_file(file.filename):
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
            file.save(file_path)
            # Process the uploaded audio file to text
            audio_text = total.process_audio(file_path)
            # Generate abstractive summarization of the text
            theme = total.abstract(audio_text)
            summary = total.abstractive_summarization(audio_text, max_chunk_length=1500, max_summary_length=300)
            os.remove(file_path)
            return render_template('headline.html', theme=theme, summary=summary)
        else:
            return 'Invalid file format'
        
    return render_template('headline.html')


if __name__ == '__main__':
    app.run(debug=True)
