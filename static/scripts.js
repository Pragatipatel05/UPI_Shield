// UPIShield - Custom JavaScript for Enhanced User Experience

document.addEventListener('DOMContentLoaded', function() {
    // Initialize page features
    initializeAnimations();
    initializeFormValidation();
    initializeTooltips();
    initializeProgressBars();
    
    // Add smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
    
    // Add loading states to forms
    setupFormLoadingStates();
    
    // Initialize file upload handling
    setupFileUpload();
    
    // Add keyboard shortcuts
    setupKeyboardShortcuts();
});

// Animate elements on scroll
function initializeAnimations() {
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, observerOptions);
    
    // Observe cards and sections
    document.querySelectorAll('.card, .fraud-card, section').forEach(el => {
        el.style.opacity = '0';
        el.style.transform = 'translateY(30px)';
        el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(el);
    });
}

// Form validation and enhancement
function initializeFormValidation() {
    const forms = document.querySelectorAll('form');
    
    forms.forEach(form => {
        form.addEventListener('submit', function(e) {
            if (!validateForm(form)) {
                e.preventDefault();
                showError('Please fill in all required fields correctly.');
                return false;
            }
            
            // Add loading state
            const submitBtn = form.querySelector('button[type="submit"]');
            if (submitBtn) {
                addLoadingState(submitBtn);
            }
        });
        
        // Real-time validation for numeric inputs
        const numericInputs = form.querySelectorAll('input[type="number"]');
        numericInputs.forEach(input => {
            input.addEventListener('input', function() {
                validateNumericInput(this);
            });
            
            input.addEventListener('blur', function() {
                validateNumericInput(this);
            });
        });
    });
}

// Validate form fields
function validateForm(form) {
    const requiredFields = form.querySelectorAll('[required]');
    let isValid = true;
    
    requiredFields.forEach(field => {
        if (!field.value.trim()) {
            markFieldInvalid(field, 'This field is required');
            isValid = false;
        } else {
            markFieldValid(field);
        }
    });
    
    return isValid;
}

// Validate numeric inputs
function validateNumericInput(input) {
    const value = parseFloat(input.value);
    const fieldName = input.previousElementSibling?.textContent || 'Field';
    
    if (isNaN(value)) {
        markFieldInvalid(input, 'Please enter a valid number');
        return false;
    }
    
    // Specific validations based on field name
    if (fieldName.includes('Hour') && (value < 0 || value > 23)) {
        markFieldInvalid(input, 'Hour must be between 0-23');
        return false;
    }
    
    if (fieldName.includes('Day') && (value < 1 || value > 31)) {
        markFieldInvalid(input, 'Day must be between 1-31');
        return false;
    }
    
    if (fieldName.includes('Month') && (value < 1 || value > 12)) {
        markFieldInvalid(input, 'Month must be between 1-12');
        return false;
    }
    
    if (fieldName.includes('Age') && (value < 0 || value > 120)) {
        markFieldInvalid(input, 'Please enter a valid age');
        return false;
    }
    
    if (fieldName.includes('Amount') && value < 0) {
        markFieldInvalid(input, 'Amount cannot be negative');
        return false;
    }
    
    markFieldValid(input);
    return true;
}

// Mark field as invalid
function markFieldInvalid(field, message) {
    field.classList.add('is-invalid');
    field.classList.remove('is-valid');
    
    // Remove existing feedback
    const existingFeedback = field.parentNode.querySelector('.invalid-feedback');
    if (existingFeedback) {
        existingFeedback.remove();
    }
    
    // Add new feedback
    const feedback = document.createElement('div');
    feedback.className = 'invalid-feedback';
    feedback.textContent = message;
    field.parentNode.appendChild(feedback);
}

// Mark field as valid
function markFieldValid(field) {
    field.classList.add('is-valid');
    field.classList.remove('is-invalid');
    
    const feedback = field.parentNode.querySelector('.invalid-feedback');
    if (feedback) {
        feedback.remove();
    }
}

// Initialize tooltips for better UX
function initializeTooltips() {
    // Add tooltips to fraud cards
    const fraudCards = document.querySelectorAll('.fraud-card');
    fraudCards.forEach(card => {
        card.setAttribute('title', 'Click to learn more about this fraud type');
        card.style.cursor = 'pointer';
        
        card.addEventListener('click', function() {
            const title = this.querySelector('h3, h5, h6')?.textContent;
            const description = this.querySelector('p')?.textContent;
            showInfoModal(title, description);
        });
    });
}

// Initialize progress bars animation
function initializeProgressBars() {
    const progressBars = document.querySelectorAll('.progress-bar');
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const progressBar = entry.target;
                const width = progressBar.style.width;
                progressBar.style.width = '0%';
                
                setTimeout(() => {
                    progressBar.style.width = width;
                    progressBar.style.transition = 'width 1.5s ease-in-out';
                }, 100);
            }
        });
    });
    
    progressBars.forEach(bar => observer.observe(bar));
}

// Setup form loading states
function setupFormLoadingStates() {
    document.querySelectorAll('form').forEach(form => {
        form.addEventListener('submit', function() {
            const submitBtn = form.querySelector('button[type="submit"]');
            if (submitBtn) {
                addLoadingState(submitBtn);
            }
        });
    });
}

// Add loading state to button
function addLoadingState(button) {
    button.disabled = true;
    const originalText = button.innerHTML;
    const spinner = '<span class="spinner-border spinner-border-sm me-2" role="status"></span>';
    
    if (button.id === 'trainBtn') {
        button.innerHTML = spinner + 'Training Model...';
    } else if (button.id === 'detectBtn') {
        button.innerHTML = spinner + 'Analyzing...';
    } else {
        button.innerHTML = spinner + 'Processing...';
    }
    
    // Store original text for potential restoration
    button.setAttribute('data-original-text', originalText);
}

// Setup file upload handling
function setupFileUpload() {
    const fileInputs = document.querySelectorAll('input[type="file"]');
    
    fileInputs.forEach(input => {
        input.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                validateFile(file, input);
                showFileInfo(file);
            }
        });
        
        // Drag and drop functionality
        const uploadArea = input.closest('.upload-area');
        if (uploadArea) {
            setupDragAndDrop(uploadArea, input);
        }
    });
}

// Validate uploaded file
function validateFile(file, input) {
    const maxSize = 10 * 1024 * 1024; // 10MB
    const allowedTypes = ['text/csv', 'application/vnd.ms-excel'];
    
    if (file.size > maxSize) {
        showError('File size must be less than 10MB');
        input.value = '';
        return false;
    }
    
    if (!allowedTypes.includes(file.type) && !file.name.endsWith('.csv')) {
        showError('Please upload a CSV file');
        input.value = '';
        return false;
    }
    
    return true;
}

// Show file information
function showFileInfo(file) {
    const fileSize = (file.size / 1024 / 1024).toFixed(2);
    showSuccess(`File selected: ${file.name} (${fileSize} MB)`);
}

// Setup drag and drop for file upload
function setupDragAndDrop(uploadArea, input) {
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, preventDefaults, false);
    });
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    ['dragenter', 'dragover'].forEach(eventName => {
        uploadArea.addEventListener(eventName, highlight, false);
    });
    
    ['dragleave', 'drop'].forEach(eventName => {
        uploadArea.addEventListener(eventName, unhighlight, false);
    });
    
    function highlight() {
        uploadArea.classList.add('drag-over');
        uploadArea.style.borderColor = 'var(--primary-color)';
        uploadArea.style.background = 'rgba(13, 110, 253, 0.1)';
    }
    
    function unhighlight() {
        uploadArea.classList.remove('drag-over');
        uploadArea.style.borderColor = '';
        uploadArea.style.background = '';
    }
    
    uploadArea.addEventListener('drop', handleDrop, false);
    
    function handleDrop(e) {
        const dt = e.dataTransfer;
        const files = dt.files;
        
        if (files.length > 0) {
            input.files = files;
            input.dispatchEvent(new Event('change'));
        }
    }
}

// Setup keyboard shortcuts
function setupKeyboardShortcuts() {
    document.addEventListener('keydown', function(e) {
        // Ctrl/Cmd + Enter to submit forms
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            const activeForm = document.querySelector('form:focus-within');
            if (activeForm) {
                const submitBtn = activeForm.querySelector('button[type="submit"]');
                if (submitBtn && !submitBtn.disabled) {
                    submitBtn.click();
                }
            }
        }
        
        // Escape to close modals
        if (e.key === 'Escape') {
            const modal = document.querySelector('.modal.show');
            if (modal) {
                const closeBtn = modal.querySelector('.btn-close');
                if (closeBtn) closeBtn.click();
            }
        }
    });
}

// Utility functions for notifications
function showSuccess(message) {
    showNotification(message, 'success');
}

function showError(message) {
    showNotification(message, 'error');
}

function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `alert alert-${type === 'error' ? 'danger' : type} alert-dismissible fade show position-fixed`;
    notification.style.cssText = `
        top: 100px;
        right: 20px;
        z-index: 9999;
        min-width: 300px;
        max-width: 400px;
    `;
    
    notification.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    document.body.appendChild(notification);
    
    // Auto remove after 5 seconds
    setTimeout(() => {
        if (notification.parentNode) {
            notification.remove();
        }
    }, 5000);
}

// Show info modal
function showInfoModal(title, description) {
    // Create modal if it doesn't exist
    let modal = document.getElementById('infoModal');
    if (!modal) {
        modal = document.createElement('div');
        modal.id = 'infoModal';
        modal.className = 'modal fade';
        modal.innerHTML = `
            <div class="modal-dialog modal-dialog-centered">
                <div class="modal-content bg-dark text-light">
                    <div class="modal-header border-secondary">
                        <h5 class="modal-title" id="infoModalTitle">${title}</h5>
                        <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal"></button>
                    </div>
                    <div class="modal-body">
                        <p id="infoModalBody">${description}</p>
                    </div>
                    <div class="modal-footer border-secondary">
                        <button type="button" class="btn btn-primary" data-bs-dismiss="modal">Got it</button>
                    </div>
                </div>
            </div>
        `;
        document.body.appendChild(modal);
    } else {
        document.getElementById('infoModalTitle').textContent = title;
        document.getElementById('infoModalBody').textContent = description;
    }
    
    // Show modal using Bootstrap
    const bsModal = new bootstrap.Modal(modal);
    bsModal.show();
}

// Enhanced security features
function sanitizeInput(input) {
    const div = document.createElement('div');
    div.textContent = input;
    return div.innerHTML;
}

// Performance monitoring
function logPerformance(action) {
    if (performance && performance.now) {
        console.log(`Action: ${action}, Time: ${performance.now().toFixed(2)}ms`);
    }
}

// Auto-save form data (for prediction form)
function setupAutoSave() {
    const predictForm = document.getElementById('predictForm');
    if (predictForm) {
        const inputs = predictForm.querySelectorAll('input');
        
        inputs.forEach(input => {
            input.addEventListener('input', function() {
                const formData = new FormData(predictForm);
                const data = Object.fromEntries(formData.entries());
                localStorage.setItem('upishield_form_data', JSON.stringify(data));
            });
        });
        
        // Restore form data on page load
        const savedData = localStorage.getItem('upishield_form_data');
        if (savedData) {
            try {
                const data = JSON.parse(savedData);
                Object.keys(data).forEach(key => {
                    const input = predictForm.querySelector(`[name="${key}"]`);
                    if (input) {
                        input.value = data[key];
                    }
                });
            } catch (e) {
                console.warn('Failed to restore form data:', e);
            }
        }
    }
}

// Call auto-save setup
setupAutoSave();

// Theme toggle (future enhancement)
function toggleTheme() {
    document.body.classList.toggle('light-theme');
    localStorage.setItem('theme', document.body.classList.contains('light-theme') ? 'light' : 'dark');
}

// Add some visual feedback for interactions
document.addEventListener('click', function(e) {
    if (e.target.classList.contains('btn')) {
        // Create ripple effect
        const ripple = document.createElement('span');
        ripple.className = 'ripple';
        ripple.style.cssText = `
            position: absolute;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.3);
            pointer-events: none;
            transform: scale(0);
            animation: ripple 0.6s linear;
        `;
        
        const rect = e.target.getBoundingClientRect();
        const size = Math.max(rect.width, rect.height);
        ripple.style.width = ripple.style.height = size + 'px';
        ripple.style.left = (e.clientX - rect.left - size / 2) + 'px';
        ripple.style.top = (e.clientY - rect.top - size / 2) + 'px';
        
        e.target.style.position = 'relative';
        e.target.style.overflow = 'hidden';
        e.target.appendChild(ripple);
        
        setTimeout(() => ripple.remove(), 600);
    }
});

// Add ripple animation CSS
const style = document.createElement('style');
style.textContent = `
    @keyframes ripple {
        to {
            transform: scale(4);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);

console.log('UPIShield JavaScript initialized successfully!');
