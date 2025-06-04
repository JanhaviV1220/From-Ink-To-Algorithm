from flask import Flask, request, jsonify, render_template, send_from_directory
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import pytesseract
from pydub import AudioSegment
import whisper
import fitz
from pdf2image import convert_from_path
import pytesseract
from PIL import Image
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words
import logging
from flask_cors import CORS
import nltk
import sys
import mysql.connector

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
db_config = {
    'user': 'root',
    'password': 'fcp@123',
    'host': 'localhost',
    'database': 'Extracted_text',
    'auth_plugin': 'mysql_native_password'  # Specify the authentication plugin
}
# Download required NLTK data
def download_nltk_data():
    try:
        # First try to find the punkt tokenizer
        try:
            nltk.data.find('tokenizers/punkt')
            logger.info("NLTK punkt tokenizer already installed")
        except LookupError:
            logger.info("Downloading NLTK punkt tokenizer...")
            nltk.download('punkt', quiet=True)
            logger.info("NLTK punkt tokenizer downloaded successfully")
            
        # Also download punkt_tab which might be needed
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            logger.info("Downloading NLTK punkt_tab...")
            nltk.download('punkt_tab', quiet=True)
            logger.info("NLTK punkt_tab downloaded successfully")
    except Exception as e:
        logger.error(f"Error downloading NLTK data: {e}")
        sys.exit(1)

# Call the download function at startup
download_nltk_data()

# Configure Tesseract path
TESSERACT_PATHS = [
    r'C:\Program Files\Tesseract-OCR\tesseract.exe',  # Default Windows path
   
    'tesseract', 
]

tesseract_found = False
for path in TESSERACT_PATHS:
    if os.path.exists(path):
        pytesseract.pytesseract.tesseract_cmd = path
        tesseract_found = True
        logger.info(f"Found Tesseract at: {path}")
        break

if not tesseract_found:
    logger.warning("Tesseract not found. Image processing will not work.")

def capture_image(image_path):
    # Capture an image using the webcam (you can also load an image from a file)
    # For simplicity, we'll load an image from a file
    img = cv2.imread(image_path, -1)
    
    # Check if the image loaded successfully
    if img is None:
        print(f"Error: Could not load image from {image_path}. Please check the path.")
        return None  # or raise an exception
    
    return img

def extract_text_from_image(img):
    if img is None:
        return ""  # Return an empty string if image is not loaded

    # Use pytesseract to extract text from the image
    text = pytesseract.image_to_string(img, lang='eng')
    return text

# Initialize Flask app
app = Flask(__name__, 
            static_folder='static',
            static_url_path='/static',
            template_folder='static')

# Configure CORS
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Configure upload folder
app.config['UPLOAD_FOLDER'] = 'uploads/'
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Allowed file extensions
ALLOWED_EXTENSIONS = {
    'image': {'png', 'jpg', 'jpeg'},
    'audio': {'mp3', 'wav', 'ogg', 'm4a'},
    'pdf': {'pdf'},
    'handwritten': {'png', 'jpg', 'jpeg'}
}

# Add a test route
@app.route('/api/test', methods=['GET'])
def test_connection():
    return jsonify({'status': 'success', 'message': 'API is working'})

def allowed_file(filename, file_type):
    """Check if file extension is allowed for the given file type."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS[file_type]

def process_image(image_path, standard_size=(1024, 768), ksize=(7, 7), lang='eng+mar'):
    """
    Process an image, apply enlarging, blurring, and text extraction via OCR.
    Supports multiple languages (English and Marathi).

    Parameters:
        image_path (str): Path to the image file.
        standard_size (tuple): Minimum dimensions for resizing (width, height).
        ksize (tuple): Kernel size for Gaussian blur.
        lang (str): Languages for OCR (e.g., 'eng+mar').

    Returns:
        str: Extracted text from the image or an error message.
    """
    try:
        # Load the image
        img = cv2.imread(image_path, -1)
        if img is None:
            return f"Error: Could not load image from {image_path}. Please check the path."

        # Get the current dimensions of the image
        height, width = img.shape[:2]

        # Check if the image is smaller than the standard size
        if width < standard_size[0] or height < standard_size[1]:
            # Calculate the scaling factors
            scale_width = standard_size[0] / width
            scale_height = standard_size[1] / height
            scale = max(scale_width, scale_height)

            # Calculate the new dimensions of the image
            new_width = int(width * scale)
            new_height = int(height * scale)
            new_dimensions = (new_width, new_height)

            # Resize the image
            img = cv2.resize(img, new_dimensions, interpolation=cv2.INTER_LINEAR)

        # Split the image into RGB planes
        rgb_planes = cv2.split(img)

        result_planes = []
        result_norm_planes = []

        # Process each plane to remove shadows
        for plane in rgb_planes:
            dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
            bg_img = cv2.medianBlur(dilated_img, 21)
            diff_img = 255 - cv2.absdiff(plane, bg_img)
            norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
            result_planes.append(diff_img)
            result_norm_planes.append(norm_img)

        # Merge the result planes
        result = cv2.merge(result_planes)
        result_norm = cv2.merge(result_norm_planes)

        # Convert the image to grayscale
        gray_img = cv2.cvtColor(result_norm, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur to the grayscale image
        blurred_img = cv2.GaussianBlur(gray_img, ksize, 0)

        # Apply thresholding to binarize the image
        _, binarized_img = cv2.threshold(blurred_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Use OCR to extract text
        extracted_text = pytesseract.image_to_string(binarized_img, lang=lang)

        return extracted_text.strip() if extracted_text else "No text detected in the image."

    except Exception as e:
        return f"Error processing image: {str(e)}"


def process_audio(audio_file_path, model_size="base"):
    """
    Convert audio to WAV format and transcribe using Whisper.
    
    Parameters:
        audio_file_path (str): Path to the input audio file.
        model_size (str): Whisper model size to use ('tiny', 'base', etc.).
    
    Returns:
        str: Transcribed text from the audio file.
    """
    try:
        # Automatically generate the output WAV file path
        base_name = os.path.splitext(audio_file_path)[0]  # Remove file extension
        output_wav_path = base_name + ".wav"  # Append .wav extension
        print(f"Output WAV Path: {output_wav_path}")

        # Convert the audio file to WAV format
        audio = AudioSegment.from_file(audio_file_path)
        audio.export(output_wav_path, format="wav")
        print(f"Audio successfully converted to WAV format: {output_wav_path}")

        # Load the Whisper model
        try:
            model = whisper.load_model(model_size)
            print(f"Loaded Whisper model: '{model_size}'")
        except Exception as model_error:
            print(f"Error loading '{model_size}' model: {model_error}")
            return "Error: Failed to load Whisper model"

        # Transcribe the WAV file
        result = model.transcribe(output_wav_path)
        text = result["text"]

        return text.strip() if text else "No text detected in the audio file"

    except Exception as e:
        print(f"Error processing audio: {e}")
        return f"Error: {str(e)}"


def process_pdf(file_path, lang='eng+mar'):
    """
    Extract text from a PDF file using PyMuPDF and Tesseract OCR for multiple languages.

    Parameters:
        file_path (str): Path to the PDF file.
        lang (str): Languages for OCR (e.g., 'eng+mar').
        tesseract_cmd (str): Path to the Tesseract executable.

    Returns:
        str: Extracted text from the PDF or an error message.
    """
    try:
        # Open the PDF file
        pdf_document = fitz.open(file_path)

        # Initialize a string to hold the extracted text
        extracted_text = ""

        # Iterate through each page
        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)

            # Try to extract text directly using PyMuPDF
            page_text = page.get_text()
            if page_text.strip():
                extracted_text += page_text
            else:
                # If no text is found, convert the page to an image and perform OCR
                pix = page.get_pixmap()
                image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                ocr_text = pytesseract.image_to_string(image, lang=lang)
                extracted_text += ocr_text

        # Close the PDF file
        pdf_document.close()

        # Check if any text was extracted
        return extracted_text.strip() if extracted_text else "No text detected in the PDF."

    except Exception as e:
        # Handle exceptions and provide feedback
        return f"Error processing PDF: {str(e)}"


def process_handwritten(file_path):
    """
    Extract text from a PDF file using PyMuPDF.

    Parameters:
        pdf_path (str): Path to the PDF file.

    Returns:
        str: Extracted text from the PDF, or an error message.
    """
    try:
        # Open the PDF file
        pdf_document = fitz.open(file_path)
        print(f"Successfully opened PDF: {file_path}")

        # Initialize a string to hold the extracted text
        extracted_text = ""

        # Iterate through each page and extract text
        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            extracted_text += page.get_text()
            print(f"Extracted text from page {page_num + 1}/{pdf_document.page_count}")

        # Close the PDF file
        pdf_document.close()

        # Check if any text was extracted
        return extracted_text.strip() if extracted_text else "No text detected in the PDF."

    except Exception as e:
        # Handle exceptions and provide feedback
        print(f"Error processing PDF: {e}")
        return f"Error processing PDF: {str(e)}"


def summarize_text(text, sentence_count=5):
    """Summarize the extracted text."""
    try:
        # Check if NLTK data is available
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            return "Error: NLTK data is not available. Please run: python -c \"import nltk; nltk.download('punkt')\""
        
        if not text.strip():
            return "No text provided for summarization"
            
        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        stemmer = Stemmer("english")
        summarizer = LsaSummarizer(stemmer)
        summarizer.stop_words = get_stop_words("english")
        
        summary = summarizer(parser.document, sentence_count)
        return " ".join([str(sentence) for sentence in summary]) if summary else "Could not generate summary"
    except Exception as e:
        logger.error(f"Error summarizing text: {e}")
        return f"Error summarizing text: {str(e)}"

def find_matching_texts(extracted_text):
    try:
        # Connect to the database
        conn = mysql.connector.connect(**db_config)
        cursor = conn.cursor()

        # Get all entries from the database first
        cursor.execute("SELECT id, extracted_text FROM ExtractedText")
        all_db_entries = cursor.fetchall()
        
        print(f"Found {len(all_db_entries)} total entries in the database")
        
        # Skip processing if no entries in database
        if not all_db_entries:
            print("No entries in the database to compare with")
            return []
            
        # Split the extracted text into lines and clean them
        lines = [line.strip() for line in extracted_text.split('\n') if line.strip()]
        
        # Filter out very short lines that might cause too many matches
        significant_lines = [line for line in lines if len(line) >= 5]
        
        print(f"Checking {len(significant_lines)} significant lines from extracted text")
        
        # Store matching entries
        matching_entries = []
        matching_ids = set()  # To track which entries we've already matched
        
        # For each database entry, check if any line from extracted text is in it
        for db_id, db_text in all_db_entries:
            if not db_text or db_id in matching_ids:
                continue
                
            db_text_lower = db_text.lower()
            
            # Check each line from extracted text
            for line in significant_lines:
                if line.lower() in db_text_lower:
                    print(f"Entry {db_id} matches line: {line[:30]}...")
                    matching_entries.append(db_text)
                    matching_ids.add(db_id)
                    break  # Found a match for this entry, move to next entry
        
        print(f"Found {len(matching_entries)} matching entries")
        
        # Close the cursor and connection
        cursor.close()
        conn.close()

        return matching_entries

    except mysql.connector.Error as err:
        logger.error(f"Database error: {err}")
        return None

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

@app.route('/api/process', methods=['POST'])
def handle_request():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        file_type = request.form.get('type', '')
        if not file_type:
            return jsonify({'error': 'No file type specified'}), 400

        if not allowed_file(file.filename, file_type):
            return jsonify({'error': 'File type not allowed'}), 400

        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        result = None
        if file_type == 'image':
            result = process_image(file_path)
        elif file_type == 'audio':
            result = process_audio(file_path)
        elif file_type == 'pdf':
            result = process_pdf(file_path)
        elif file_type == 'handwritten':
            result = process_handwritten(file_path)

        if result:
            # Summarize the text
            summary = summarize_text(result)
            
            # Find similar content in the database
            similar_content = find_matching_texts(result) or []
            
            return jsonify({
                'success': True,
                'text': result,
                'summary': summary,
                'similar_content': similar_content
            })
        else:
            return jsonify({'error': 'Failed to process file'}), 500

    except Exception as e:
        logging.error(f"Error processing file: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/summarize', methods=['POST'])
def summarize():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400

        text = data['text']
        summary = summarize_text(text)
        
        return jsonify({
            'success': True,
            'summary': summary
        })

    except Exception as e:
        logger.error(f"Error in summarize endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/check_database', methods=['POST'])
def check_database():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400

        text = data['text']
        
        # Find similar content in the database
        similar_content = find_matching_texts(text) or []
        
        return jsonify({
            'success': True,
            'similar_content': similar_content
        })

    except Exception as e:
        logger.error(f"Error in check_database endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    try:
        # Test database connection
        try:
            conn = mysql.connector.connect(**db_config)
            if conn.is_connected():
                print("Successfully connected to MySQL database")
                cursor = conn.cursor()
                
                # Check if the required table exists
                cursor.execute("SHOW TABLES LIKE 'ExtractedText'")
                if cursor.fetchone() is None:
                    print("Creating ExtractedText table...")
                    cursor.execute("""
                    CREATE TABLE IF NOT EXISTS ExtractedText (
                        id INT AUTO_INCREMENT PRIMARY KEY,
                        extracted_text TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                    """)
                    conn.commit()
                    print("ExtractedText table created successfully")
                else:
                    print("ExtractedText table already exists")
                
                cursor.close()
            conn.close()
        except mysql.connector.Error as err:
            print(f"Error connecting to MySQL database: {err}")
            print("The application will start but database features may not work properly.")
            
        print("Starting Flask application...")
        app.run(host='0.0.0.0', port=5000, debug=True)
    except Exception as e:
        print(f"Error starting application: {e}")