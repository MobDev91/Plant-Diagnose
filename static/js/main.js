// PlantDiagnose - Main JavaScript

document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

function initializeApp() {
    // Initialize animations
    initializeAnimations();
    
    // Initialize tooltips if Bootstrap is available
    initializeTooltips();
    
    // Initialize smooth scrolling
    initializeSmoothScrolling();
    
    // Initialize form enhancements
    initializeFormEnhancements();
}

function initializeAnimations() {
    // Stagger fade-in animations
    const elements = document.querySelectorAll('.fade-in-up');
    elements.forEach((el, index) => {
        el.style.animationDelay = (index * 0.1) + 's';
    });
    
    // Add intersection observer for scroll-triggered animations
    if ('IntersectionObserver' in window) {
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('fade-in-up');
                }
            });
        });
        
        document.querySelectorAll('.animate-on-scroll').forEach(el => {
            observer.observe(el);
        });
    }
}

function initializeTooltips() {
    // Initialize Bootstrap tooltips if available
    if (typeof bootstrap !== 'undefined' && bootstrap.Tooltip) {
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }
}

function initializeSmoothScrolling() {
    // Smooth scroll for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                e.preventDefault();
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
}

function initializeFormEnhancements() {
    // Add floating label effect
    const inputs = document.querySelectorAll('.form-control-modern');
    inputs.forEach(input => {
        // Add focus/blur effects
        input.addEventListener('focus', function() {
            this.parentElement.classList.add('focused');
        });
        
        input.addEventListener('blur', function() {
            if (!this.value) {
                this.parentElement.classList.remove('focused');
            }
        });
    });
}

// Global utility functions
function showNotification(message, type = 'info') {
    // Simple notification system
    const notification = document.createElement('div');
    notification.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
    notification.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
    notification.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    document.body.appendChild(notification);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (notification.parentElement) {
            notification.remove();
        }
    }, 5000);
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Global loading functions
function showGlobalLoading(message = 'Loading...') {
    let overlay = document.querySelector('.loading-overlay');
    if (!overlay) {
        overlay = document.createElement('div');
        overlay.className = 'loading-overlay';
        overlay.innerHTML = `
            <div class="loading-content text-center">
                <div class="spinner-border mb-3" style="color: var(--primary); width: 3rem; height: 3rem;"></div>
                <h4 class="fw-bold">${message}</h4>
            </div>
        `;
        document.body.appendChild(overlay);
    }
    overlay.classList.add('show');
}

function hideGlobalLoading() {
    const overlay = document.querySelector('.loading-overlay');
    if (overlay) {
        overlay.classList.remove('show');
    }
}

// Error handling
window.addEventListener('error', function(e) {
    console.error('Global error:', e.error);
    // Don't show error notifications in production
    if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
        showNotification('An error occurred. Check console for details.', 'danger');
    }
});

// Performance monitoring
if ('performance' in window) {
    window.addEventListener('load', function() {
        setTimeout(() => {
            const perfData = performance.getEntriesByType('navigation')[0];
            const loadTime = perfData.loadEventEnd - perfData.loadEventStart;
            console.log(`Page load time: ${loadTime}ms`);
        }, 0);
    });
}

// Export functions for use in other scripts
window.PlantDiagnose = {
    showNotification,
    formatFileSize,
    showGlobalLoading,
    hideGlobalLoading
};
