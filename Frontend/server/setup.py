import os
import sys
import subprocess
import nltk

def check_tesseract():
    """Check if Tesseract is installed."""
    try:
        subprocess.run(['tesseract', '--version'], capture_output=True, text=True)
        print("✓ Tesseract OCR is installed")
        return True
    except FileNotFoundError:
        print("✗ Tesseract OCR is not installed")
        print("Please install Tesseract OCR from: https://github.com/UB-Mannheim/tesseract/wiki")
        print("Make sure to add it to your system PATH during installation")
        return False

def check_nltk_data():
    """Check and download required NLTK data."""
    try:
        nltk.data.find('tokenizers/punkt')
        print("✓ NLTK punkt tokenizer is installed")
        return True
    except LookupError:
        print("Downloading NLTK punkt tokenizer...")
        try:
            nltk.download('punkt')
            print("✓ NLTK punkt tokenizer downloaded successfully")
            return True
        except Exception as e:
            print(f"✗ Failed to download NLTK data: {e}")
            return False

def main():
    print("Checking dependencies...")
    
    tesseract_ok = check_tesseract()
    nltk_ok = check_nltk_data()
    
    if tesseract_ok and nltk_ok:
        print("\nAll dependencies are properly installed!")
        print("You can now run the Flask application with: python app3.py")
    else:
        print("\nSome dependencies are missing. Please install them before running the application.")
        sys.exit(1)

if __name__ == "__main__":
    main() 