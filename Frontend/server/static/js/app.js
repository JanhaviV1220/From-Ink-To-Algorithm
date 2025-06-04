document.addEventListener('DOMContentLoaded', () => {
    const uploadForm = document.getElementById('uploadForm');
    const fileInput = document.getElementById('fileInput');
    const loading = document.getElementById('loading');
    const results = document.getElementById('results');
    const error = document.getElementById('error');
    const extractedText = document.getElementById('extractedText');
    
    let selectedFileType = 'image'; // Default file type
    let currentText = ''; // Store the current extracted text

    // Get all action buttons using a more compatible selector approach
    const actionButtons = document.querySelectorAll('button');
    const generateSummaryBtn = Array.from(actionButtons).find(button => 
        button.querySelector('.material-symbols-outlined')?.textContent === 'summarize'
    );
    const checkDatabaseBtn = Array.from(actionButtons).find(button => 
        button.querySelector('.material-symbols-outlined')?.textContent === 'database'
    );
    const copyBtn = Array.from(actionButtons).find(button => 
        button.querySelector('.material-symbols-outlined')?.textContent === 'content_copy'
    );
    const downloadBtn = Array.from(actionButtons).find(button => 
        button.querySelector('.material-symbols-outlined')?.textContent === 'download'
    );

    // Setup file type buttons
    const fileTypeButtons = document.querySelectorAll('.flex.flex-wrap.gap-4.mb-6 button');
    fileTypeButtons.forEach(button => {
        button.addEventListener('click', (e) => {
            e.preventDefault();
            // Remove active state from all buttons
            fileTypeButtons.forEach(btn => {
                btn.classList.remove('bg-primary-200');
                btn.classList.add('bg-primary-100');
            });
            // Add active state to clicked button
            button.classList.remove('bg-primary-100');
            button.classList.add('bg-primary-200');
            // Set selected file type based on button text content
            const buttonText = button.querySelector('.material-symbols-outlined')?.nextSibling?.textContent.trim().toLowerCase() || '';
            selectedFileType = buttonText;
            // Update file input accept attribute
            updateFileInputAccept(buttonText);
        });
    });

    // Update file input accept attribute based on selected type
    function updateFileInputAccept(type) {
        const acceptMap = {
            image: 'image/*',
            audio: 'audio/*',
            pdf: '.pdf,application/pdf'
        };
        fileInput.setAttribute('accept', acceptMap[type] || '*/*');
    }

    // Add file input validation
    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            const allowedTypes = {
                image: ['image/jpeg', 'image/png', 'image/jpg', 'image/gif'],
                audio: ['audio/mpeg', 'audio/wav', 'audio/ogg', 'audio/mp3', 'audio/m4a'],
                pdf: ['application/pdf']
            };

            const fileType = file.type.toLowerCase();
            const allowedExtensions = allowedTypes[selectedFileType] || [];

            if (!allowedExtensions.includes(fileType)) {
                // Special check for PDF files with no mime type
                if (selectedFileType === 'pdf' && file.name.toLowerCase().endsWith('.pdf')) {
                    handleFileSelection(file);
                    return;
                }
                showError(`Please select a valid ${selectedFileType} file`);
                fileInput.value = '';
                return;
            }

            // Show the results container when a valid file is selected
            handleFileSelection(file);
        }
    });

    // Handle file selection
    function handleFileSelection(file) {
        results.classList.remove('hidden');
        extractedText.textContent = 'Your extracted text will appear here once processed.';
        
        // Start file processing
        processFile(file);
    }

    // Process the file
    async function processFile(file) {
        try {
            const formData = new FormData();
            formData.append('file', file);
            formData.append('type', selectedFileType);

            // Show loading state
            loading.classList.remove('hidden');
            results.classList.add('hidden');
            error.classList.add('hidden');

            const response = await fetch('/api/process', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (data.error) {
                showError(data.error);
                return;
            }

            // Hide loading and show results
            loading.classList.add('hidden');
            results.classList.remove('hidden');
            
            // Store and display results
            currentText = data.text || 'No text extracted';
            extractedText.textContent = currentText;
            
        } catch (err) {
            showError('An error occurred while processing the file');
            console.error('Error:', err);
        }
    }

    // Generate Summary functionality
    if (generateSummaryBtn) {
        generateSummaryBtn.addEventListener('click', async () => {
            if (!currentText || currentText === 'No text extracted') {
                showError('Please process a file first to generate a summary');
                return;
            }

            try {
                // Show loading state
                loading.classList.remove('hidden');
                results.classList.add('hidden');

                const response = await fetch('/api/summarize', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ text: currentText })
                });

                const data = await response.json();

                // Hide loading state
                loading.classList.add('hidden');
                results.classList.remove('hidden');

                if (data.error) {
                    showError(data.error);
                    return;
                }

                // Display the summary
                extractedText.textContent = data.summary || 'Could not generate summary';

            } catch (err) {
                showError('An error occurred while generating the summary');
                console.error('Error:', err);
            }
        });
    }

    // Check Database functionality
    if (checkDatabaseBtn) {
        checkDatabaseBtn.addEventListener('click', async () => {
            if (!currentText || currentText === 'No text extracted') {
                showError('Please process a file first to check in database');
                return;
            }

            try {
                // Show loading state
                loading.classList.remove('hidden');
                results.classList.add('hidden');

                const response = await fetch('/api/check_database', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ 
                        text: currentText
                    })
                });

                const data = await response.json();

                // Hide loading state
                loading.classList.add('hidden');
                results.classList.remove('hidden');

                if (data.error) {
                    showError(data.error);
                    return;
                }

                // Display matching texts from database
                if (data.similar_content && data.similar_content.length > 0) {
                    extractedText.textContent = 'Similar content found in database:\n\n' + 
                                              data.similar_content.join('\n\n');
                } else {
                    extractedText.textContent = 'No similar content found in database.';
                }

            } catch (err) {
                showError('An error occurred while checking the database');
                console.error('Error:', err);
            }
        });
    }

    // Copy functionality
    if (copyBtn) {
        copyBtn.addEventListener('click', async () => {
            const textToCopy = extractedText.textContent;
            if (!textToCopy || textToCopy === 'Your extracted text will appear here once processed.') {
                showError('No text available to copy');
                return;
            }

            try {
                await navigator.clipboard.writeText(textToCopy);
                // Show success message
                const originalText = copyBtn.innerHTML;
                copyBtn.innerHTML = '<span class="material-symbols-outlined text-green-500">check</span>';
                setTimeout(() => {
                    copyBtn.innerHTML = originalText;
                }, 2000);
            } catch (err) {
                showError('Failed to copy text to clipboard');
            }
        });
    }

    // Download functionality
    if (downloadBtn) {
        downloadBtn.addEventListener('click', () => {
            const textToDownload = extractedText.textContent;
            if (!textToDownload || textToDownload === 'Your extracted text will appear here once processed.') {
                showError('No text available to download');
                return;
            }

            const blob = new Blob([textToDownload], { type: 'text/plain' });
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'extracted_text.txt';
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
        });
    }

    // Setup drag and drop
    uploadForm.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadForm.classList.add('border-primary-400');
    });

    uploadForm.addEventListener('dragleave', (e) => {
        e.preventDefault();
        uploadForm.classList.remove('border-primary-400');
    });

    uploadForm.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadForm.classList.remove('border-primary-400');
        
        const file = e.dataTransfer.files[0];
        if (file) {
            fileInput.files = e.dataTransfer.files;
            const event = new Event('change');
            fileInput.dispatchEvent(event);
        }
    });

    function showError(message) {
        loading.classList.add('hidden');
        results.classList.add('hidden');
        error.classList.remove('hidden');
        error.querySelector('.error-message').textContent = message;
        
        // Auto-hide error after 5 seconds
        setTimeout(() => {
            error.classList.add('hidden');
        }, 5000);
    }

    // Test API connection on page load
    fetch('/api/test')
        .then(response => response.json())
        .then(data => {
            if (data.status !== 'success') {
                showError('Unable to connect to the server');
            }
        })
        .catch(() => {
            showError('Unable to connect to the server');
        });
}); 