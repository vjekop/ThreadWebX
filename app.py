from flask import Flask, request, jsonify, send_file, render_template_string, render_template
from flask_cors import CORS
import os
import tempfile
import uuid
import json
from datetime import datetime
import threading
import time
import subprocess

# YouTube and AI imports
from pytube import YouTube
from openai import OpenAI
import whisper

# Firebase imports
import firebase_admin
from firebase_admin import credentials, firestore, auth, storage

# Initialize Flask with proper template folder
app = Flask(__name__, template_folder='.')  # Look for templates in current directory
CORS(app)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Initialize Firebase
if not firebase_admin._apps:
    try:
        # Try to load from file first
        cred = credentials.Certificate('firebase-config.json')
    except:
        # Fallback to environment variables for Railway
        firebase_config = {
            "type": "service_account",
            "project_id": os.getenv('FIREBASE_PROJECT_ID'),
            "private_key_id": os.getenv('FIREBASE_PRIVATE_KEY_ID'),
            "private_key": os.getenv('FIREBASE_PRIVATE_KEY', '').replace('\\n', '\n'),
            "client_email": os.getenv('FIREBASE_CLIENT_EMAIL'),
            "client_id": os.getenv('FIREBASE_CLIENT_ID'),
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_x509_cert_url": os.getenv('FIREBASE_CLIENT_CERT_URL')
        }
        cred = credentials.Certificate(firebase_config)
    
    firebase_admin.initialize_app(cred, {
        'storageBucket': 'gs://trendsaas-29b27.firebasestorage.appspot.com'
    })

db = firestore.client()

# Initialize OpenAI with NEW API format
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Simple in-memory job storage
jobs = {}

def verify_token(f):
    def decorator(*args, **kwargs):
        token = request.headers.get('Authorization', '').replace('Bearer ', '')
        if not token:
            return jsonify({'error': 'No token provided'}), 401
        try:
            decoded_token = auth.verify_id_token(token)
            request.user = decoded_token
            return f(*args, **kwargs)
        except Exception as e:
            return jsonify({'error': 'Invalid token'}), 401
    decorator.__name__ = f.__name__
    return decorator

def transcribe_audio_whisper_api(audio_path):
    """Transcribe audio using OpenAI Whisper API"""
    try:
        with open(audio_path, 'rb') as audio_file:
            response = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )
        return response
    except Exception as e:
        print(f"Whisper API error: {e}")
        # Fallback to local whisper if API fails
        return transcribe_audio_local(audio_path)

def transcribe_audio_local(audio_path):
    """Fallback: Transcribe audio using local Whisper"""
    try:
        model = whisper.load_model("base")
        result = model.transcribe(audio_path)
        return result["text"]
    except Exception as e:
        print(f"Local Whisper error: {e}")
        return "Error: Could not transcribe audio"

def generate_threads_ai(content, title=None):
    """Generate Twitter threads using OpenAI - UPDATED API FORMAT"""
    prompt = f"""Create 2 engaging Twitter threads from this content. Make them viral and shareable.

Content: {content}
{f"Title: {title}" if title else ""}

Return JSON format:
[
  {{
    "title": "Thread title",
    "tweets": ["Tweet 1 with hook 🧵", "1/ First point...", "2/ Second point...", "Final tweet with CTA"],
    "engagement": "High"
  }}
]

Rules:
- First tweet must grab attention
- Use emojis and numbered format (1/, 2/, etc.)
- 5-10 tweets per thread
- End with call-to-action
- Keep each tweet under 280 characters"""

    try:
        # NEW OPENAI API FORMAT
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1500
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"OpenAI error: {e}")
        # Fallback simple thread
        sentences = content.split('. ')[:8]
        tweets = [f"🧵 {title or 'Key insights'} - Thread:"]
        for i, sentence in enumerate(sentences, 1):
            if sentence.strip():
                tweets.append(f"{i}/ {sentence.strip()}.")
        tweets.append("What do you think? Drop your thoughts below! 👇")
        
        return [{
            "title": title or "Generated Thread",
            "tweets": tweets,
            "engagement": "Medium"
        }]

def download_youtube_audio(url):
    """Download YouTube audio with better error handling"""
    try:
        yt = YouTube(url)
        # Get audio stream with fallback options
        audio_stream = (
            yt.streams.filter(only_audio=True, file_extension='mp4').first() or
            yt.streams.filter(only_audio=True).first() or
            yt.streams.filter(adaptive=True, file_extension='mp4').first()
        )
        
        if not audio_stream:
            raise Exception("No suitable audio stream found")
        
        # Create temp file
        temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        temp_file.close()
        
        # Download
        audio_stream.download(filename=temp_file.name)
        
        return temp_file.name, yt.title
    except Exception as e:
        print(f"YouTube download error: {e}")
        raise

def extract_audio_from_video(video_path):
    """Extract audio from video file using ffmpeg"""
    try:
        audio_path = video_path.replace('.mp4', '.wav').replace('.mov', '.wav').replace('.avi', '.wav')
        
        # Use ffmpeg to extract audio
        cmd = [
            'ffmpeg', '-i', video_path, 
            '-ab', '160k', '-ac', '2', '-ar', '44100', '-vn', 
            audio_path, '-y'  # -y to overwrite output files
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"FFmpeg error: {result.stderr}")
            # Try alternative format
            audio_path = video_path.replace('.mp4', '.mp3').replace('.mov', '.mp3').replace('.avi', '.mp3')
            cmd = [
                'ffmpeg', '-i', video_path, 
                '-ab', '160k', '-ac', '2', '-ar', '44100', '-vn', 
                audio_path, '-y'
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise Exception(f"FFmpeg failed: {result.stderr}")
        
        return audio_path
    except Exception as e:
        print(f"Audio extraction error: {e}")
        raise

def process_job_background(job_id, job_type, data):
    """Background processing function"""
    def run():
        try:
            jobs[job_id]['status'] = 'processing'
            jobs[job_id]['progress'] = 20
            
            if job_type == 'youtube':
                # Download YouTube video
                jobs[job_id]['progress'] = 30
                audio_path, title = download_youtube_audio(data['url'])
                
                jobs[job_id]['progress'] = 60
                # Transcribe using Whisper API
                transcript = transcribe_audio_whisper_api(audio_path)
                
                jobs[job_id]['progress'] = 80
                # Generate threads
                threads = generate_threads_ai(transcript, title)
                
                jobs[job_id].update({
                    'status': 'completed',
                    'progress': 100,
                    'result': threads
                })
                
                # Cleanup
                if os.path.exists(audio_path):
                    os.unlink(audio_path)
                    
            elif job_type == 'video':
                # Process uploaded video
                file_path = data['file_path']
                jobs[job_id]['progress'] = 40
                
                # Extract audio
                audio_path = extract_audio_from_video(file_path)
                jobs[job_id]['progress'] = 60
                
                # Transcribe
                transcript = transcribe_audio_whisper_api(audio_path)
                jobs[job_id]['progress'] = 80
                
                # Generate threads
                threads = generate_threads_ai(transcript)
                
                jobs[job_id].update({
                    'status': 'completed',
                    'progress': 100,
                    'result': threads
                })
                
                # Cleanup
                if os.path.exists(file_path):
                    os.unlink(file_path)
                if os.path.exists(audio_path):
                    os.unlink(audio_path)
                
            elif job_type == 'article':
                # Process article text
                jobs[job_id]['progress'] = 50
                
                threads = generate_threads_ai(data['text'])
                
                jobs[job_id].update({
                    'status': 'completed',
                    'progress': 100,
                    'result': threads
                })
                
        except Exception as e:
            print(f"Processing error: {e}")
            jobs[job_id].update({
                'status': 'failed',
                'error': str(e)
            })
    
    thread = threading.Thread(target=run)
    thread.start()

# Routes
@app.route('/')
def index():
    # Try multiple approaches to find the HTML file
    possible_paths = [
        'index.html',
        './index.html',
        os.path.join(os.path.dirname(__file__), 'index.html'),
        os.path.join(os.getcwd(), 'index.html')
    ]
    
    # Debug info
    print(f"Current working directory: {os.getcwd()}")
    print(f"Script directory: {os.path.dirname(__file__)}")
    print(f"Files in current directory: {os.listdir('.')}")
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    print(f"Successfully loaded HTML from: {path}")
                    return content
            except Exception as e:
                print(f"Error reading {path}: {e}")
                continue
    
    # If we can't find the HTML file, try using Flask's render_template
    try:
        return render_template('index.html')
    except Exception as e:
        print(f"Template rendering failed: {e}")
    
    # Final fallback - return a functional HTML page
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>ThreadAI - Transform Content into Viral Threads</title>
        <script src="https://www.gstatic.com/firebasejs/10.5.0/firebase-app-compat.js"></script>
        <script src="https://www.gstatic.com/firebasejs/10.5.0/firebase-auth-compat.js"></script>
        <script src="https://cdn.tailwindcss.com"></script>
        <style>
            body { background: linear-gradient(-45deg, #0a0a0f, #1a1a2e, #16213e, #0f3460); min-height: 100vh; color: white; }
            .glass { background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(20px); border: 1px solid rgba(255, 255, 255, 0.2); }
        </style>
    </head>
    <body class="flex items-center justify-center min-h-screen p-6">
        <div class="glass rounded-2xl p-8 max-w-md text-center">
            <h1 class="text-4xl font-bold mb-4">ThreadAI</h1>
            <p class="mb-6">The HTML file couldn't be loaded. Please check the deployment.</p>
            <div class="text-sm text-gray-300">
                <p>Debug info:</p>
                <p>API is running at: <a href="/health" class="text-blue-400">/health</a></p>
            </div>
        </div>
    </body>
    </html>
    """

# Debug route to check file system
@app.route('/debug')
def debug():
    debug_info = {
        'cwd': os.getcwd(),
        'script_dir': os.path.dirname(__file__),
        'files_in_cwd': os.listdir('.'),
        'files_in_script_dir': os.listdir(os.path.dirname(__file__)) if os.path.dirname(__file__) else 'N/A',
        'index_exists': os.path.exists('index.html'),
        'env_vars': {
            'PORT': os.environ.get('PORT'),
            'RAILWAY_ENVIRONMENT': os.environ.get('RAILWAY_ENVIRONMENT'),
        }
    }
    return jsonify(debug_info)

@app.route('/api/upload/youtube', methods=['POST'])
@verify_token
def process_youtube():
    data = request.get_json()
    youtube_url = data.get('url')
    
    if not youtube_url:
        return jsonify({'error': 'YouTube URL required'}), 400
    
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        'user_id': request.user['uid'],
        'status': 'pending',
        'progress': 0,
        'created_at': datetime.now().isoformat()
    }
    
    process_job_background(job_id, 'youtube', {'url': youtube_url})
    
    return jsonify({'job_id': job_id, 'message': 'Processing started'})

@app.route('/api/upload/video', methods=['POST'])
@verify_token
def upload_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Save file temporarily
    job_id = str(uuid.uuid4())
    filename = f"{job_id}_{file.filename}"
    file_path = os.path.join(tempfile.gettempdir(), filename)
    file.save(file_path)
    
    jobs[job_id] = {
        'user_id': request.user['uid'],
        'status': 'pending',
        'progress': 0,
        'created_at': datetime.now().isoformat()
    }
    
    process_job_background(job_id, 'video', {'file_path': file_path})
    
    return jsonify({'job_id': job_id, 'message': 'Processing started'})

@app.route('/api/upload/article', methods=['POST'])
@verify_token
def process_article():
    data = request.get_json()
    article_text = data.get('text')
    
    if not article_text:
        return jsonify({'error': 'Article text required'}), 400
    
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        'user_id': request.user['uid'],
        'status': 'pending',
        'progress': 0,
        'created_at': datetime.now().isoformat()
    }
    
    process_job_background(job_id, 'article', {'text': article_text})
    
    return jsonify({'job_id': job_id, 'message': 'Processing started'})

@app.route('/api/job/<job_id>/status', methods=['GET'])
@verify_token
def get_job_status(job_id):
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = jobs[job_id]
    if job['user_id'] != request.user['uid']:
        return jsonify({'error': 'Unauthorized'}), 403
    
    return jsonify(job)

@app.route('/api/job/<job_id>/result', methods=['GET'])
@verify_token
def get_job_result(job_id):
    if job_id not in jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = jobs[job_id]
    if job['user_id'] != request.user['uid']:
        return jsonify({'error': 'Unauthorized'}), 403
    
    if job['status'] != 'completed':
        return jsonify({'error': 'Job not completed'}), 400
    
    return jsonify({'threads': job['result']})

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy', 
        'message': 'Thread Generator API is running',
        'files_found': os.listdir('.'),
        'index_exists': os.path.exists('index.html')
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
