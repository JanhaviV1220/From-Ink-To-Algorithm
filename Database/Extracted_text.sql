create database Extracted_text;
use Extracted_text;
CREATE TABLE ExtractedText (
    id INT AUTO_INCREMENT PRIMARY KEY,
    source VARCHAR(255),
    author VARCHAR(255),
    extracted_text TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);