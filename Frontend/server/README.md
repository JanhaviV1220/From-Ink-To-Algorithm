# Text Extraction Tool

A Flask web application for extracting and processing text from various file types, including images, audio, PDFs, and handwritten documents.

## Features

- **Text Extraction** from multiple file formats:
  - Images (PNG, JPG, JPEG)
  - Audio files (MP3, WAV, OGG, M4A)
  - PDF documents
  - Handwritten documents (images)
- **Database Storage** of extracted text
- **Text Search** functionality
- **Text Summarization** using LSA algorithm

## Requirements

- Python 3.8+
- MySQL Server
- Tesseract OCR

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd text-extraction-tool
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   ```

3. Activate the virtual environment:
   - Windows:
     ```
     venv\Scripts\activate
     ```
   - macOS/Linux:
     ```
     source venv/bin/activate
     ```

4. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

5. Install Tesseract OCR:
   - Windows: Download and install from [https://github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki)
   - macOS: `brew install tesseract`
   - Ubuntu: `sudo apt install tesseract-ocr`

6. Create a MySQL database:
   ```sql
   CREATE DATABASE Extracted_text;
   ```

7. Update the database configuration in `app3.py` if needed:
   ```python
   db_config = {
       'user': 'your_user',
       'password': 'your_password',
       'host': 'localhost',
       'database': 'Extracted_text',
       'auth_plugin': 'mysql_native_password'
   }
   ```

## Usage

1. Start the application:
   ```
   python app3.py
   ```

2. Open your web browser and navigate to:
   ```
   http://localhost:5000
   ```

3. Use the web interface to:
   - Upload and extract text from files
   - Search for previously extracted text
   - Summarize text

## Troubleshooting

### Common issues:

1. **Tesseract not found**: Make sure Tesseract is installed and properly added to your system PATH.

2. **MySQL connection errors**: Check your MySQL credentials and ensure the MySQL server is running.

3. **Audio processing errors**: Whisper requires significant computational resources. On slower machines, audio processing may take longer or fail.

4. **Missing directories**: The application will create the necessary `uploads` directory, but if permissions are an issue, create it manually and ensure it's writable.

## License

[MIT License](LICENSE) 