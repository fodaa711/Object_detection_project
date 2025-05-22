/**
 * Main JavaScript file for YOLOv8 Object Detection web interface
 * Handles file uploads, image previews, and interactive elements
 */

document.addEventListener('DOMContentLoaded', function() {
    // File upload preview functionality
    const fileInput = document.getElementById('file-input');
    const filePlaceholder = document.getElementById('file-upload-placeholder');
    const filePreview = document.getElementById('file-upload-preview');
    const previewImage = document.getElementById('preview-image');
    const removeButton = document.getElementById('remove-image');
    
    if (fileInput) {
        fileInput.addEventListener('change', function(e) {
            if (e.target.files.length > 0) {
                const file = e.target.files[0];
                
                // Check if file is an image
                if (!file.type.match('image.*')) {
                    alert('Please select an image file (JPG, PNG, GIF)');
                    return;
                }
                
                // Display preview
                const reader = new FileReader();
                reader.onload = function(e) {
                    previewImage.src = e.target.result;
                    filePlaceholder.style.display = 'none';
                    filePreview.style.display = 'block';
                };
                reader.readAsDataURL(file);
            }
        });
        
        // Drag and drop functionality
        const dropArea = document.querySelector('.file-upload');
        
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropArea.addEventListener(eventName, preventDefaults, false);
        });
        
        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }
        
        ['dragenter', 'dragover'].forEach(eventName => {
            dropArea.addEventListener(eventName, highlight, false);
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            dropArea.addEventListener(eventName, unhighlight, false);
        });
        
        function highlight() {
            dropArea.classList.add('highlight');
        }
        
        function unhighlight() {
            dropArea.classList.remove('highlight');
        }
        
        dropArea.addEventListener('drop', handleDrop, false);
        
        function handleDrop(e) {
            const dt = e.dataTransfer;
            const files = dt.files;
            
            if (files.length > 0) {
                fileInput.files = files;
                
                // Trigger change event
                const event = new Event('change');
                fileInput.dispatchEvent(event);
            }
        }
        
        // Remove image button
        if (removeButton) {
            removeButton.addEventListener('click', function() {
                fileInput.value = '';
                previewImage.src = '#';
                filePlaceholder.style.display = 'block';
                filePreview.style.display = 'none';
            });
        }
    }
    
    // Confidence slider functionality
    const confidenceSlider = document.getElementById('confidence');
    const confidenceValue = document.getElementById('confidence-value');
    
    if (confidenceSlider && confidenceValue) {
        confidenceSlider.addEventListener('input', function() {
            confidenceValue.textContent = this.value;
        });
    }
    
    // Fullscreen modal functionality
    const fullscreenBtn = document.getElementById('fullscreen-btn');
    const modal = document.getElementById('fullscreen-modal');
    const modalImg = document.getElementById('modal-image');
    const closeModal = document.querySelector('.close-modal');
    const resultImage = document.getElementById('result-image');
    
    if (fullscreenBtn && modal && modalImg && closeModal && resultImage) {
        fullscreenBtn.addEventListener('click', function() {
            modal.style.display = 'block';
            modalImg.src = resultImage.src;
        });
        
        closeModal.addEventListener('click', function() {
            modal.style.display = 'none';
        });
        
        window.addEventListener('click', function(e) {
            if (e.target === modal) {
                modal.style.display = 'none';
            }
        });
    }
    
    // Form validation
    const uploadForm = document.getElementById('upload-form');
    
    if (uploadForm) {
        uploadForm.addEventListener('submit', function(e) {
            if (fileInput.files.length === 0) {
                e.preventDefault();
                alert('Please select an image file');
                return false;
            }
            
            // Show loading state
            const submitBtn = uploadForm.querySelector('button[type="submit"]');
            if (submitBtn) {
                submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
                submitBtn.disabled = true;
            }
            
            return true;
        });
    }
    
    // Add highlight class to file upload on click
    if (filePlaceholder) {
        filePlaceholder.addEventListener('click', function() {
            dropArea.classList.add('highlight');
            setTimeout(function() {
                dropArea.classList.remove('highlight');
            }, 300);
        });
    }
});

// Add custom CSS class for drag and drop highlight
document.head.insertAdjacentHTML('beforeend', `
    <style>
        .file-upload.highlight {
            border-color: #3498db;
            background-color: rgba(52, 152, 219, 0.05);
        }
    </style>
`);
