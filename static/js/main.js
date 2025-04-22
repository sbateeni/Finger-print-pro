document.addEventListener('DOMContentLoaded', function() {
    const uploadAreas = document.querySelectorAll('.upload-area');
    const fileInputs = document.querySelectorAll('.file-input');
    const previewImages = document.querySelectorAll('.preview-area img');
    const uploadTexts = document.querySelectorAll('.upload-text');
    const compareBtn = document.getElementById('compare-btn');
    const resultsDiv = document.getElementById('results');
    const errorMessage = document.getElementById('error-message');
    const progressBar = document.querySelector('.progress-bar');
    const matchScore = document.getElementById('match-score');
    const matchingPoints = document.getElementById('matching-points');
    const processingStatus = document.getElementById('processing-status');
    const canvas = document.getElementById('matching-canvas');
    let ctx = null;

    // Initialize canvas if it exists
    if (canvas) {
        ctx = canvas.getContext('2d');
        // Set canvas size
        function resizeCanvas() {
            const container = canvas.parentElement;
            canvas.width = container.offsetWidth;
            canvas.height = container.offsetHeight;
        }
        
        window.addEventListener('resize', resizeCanvas);
        resizeCanvas();
    }

    // Handle file selection
    fileInputs.forEach((input, index) => {
        input.addEventListener('change', function(e) {
            handleFileSelect(e.target.files[0], index);
        });
    });

    // Handle file selection
    function handleFileSelect(file, index) {
        if (file && file.type.startsWith('image/')) {
            const reader = new FileReader();
            reader.onload = function(e) {
                previewImages[index].src = e.target.result;
                previewImages[index].style.display = 'block';
                uploadTexts[index].style.display = 'none';
                uploadAreas[index].classList.add('has-image');
            };
            reader.readAsDataURL(file);
            updateCompareButton();
        } else if (file) {
            showError('الرجاء اختيار ملف صورة صالح');
        }
    }

    // Handle drag and drop
    uploadAreas.forEach((area, index) => {
        area.addEventListener('dragover', function(e) {
            e.preventDefault();
            e.stopPropagation();
            area.classList.add('dragover');
        });

        area.addEventListener('dragleave', function(e) {
            e.preventDefault();
            e.stopPropagation();
            area.classList.remove('dragover');
        });

        area.addEventListener('drop', function(e) {
            e.preventDefault();
            e.stopPropagation();
            area.classList.remove('dragover');
            
            const file = e.dataTransfer.files[0];
            if (file && file.type.startsWith('image/')) {
                handleFileSelect(file, index);
            } else {
                showError('الرجاء إسقاط ملف صورة صالح');
            }
        });

        // Handle click to upload
        area.addEventListener('click', function() {
            fileInputs[index].click();
        });
    });

    // Update compare button state
    function updateCompareButton() {
        const bothFilesSelected = Array.from(fileInputs).every(input => input.files.length > 0);
        if (compareBtn) {
            compareBtn.disabled = !bothFilesSelected;
        }
    }

    // Show error message
    function showError(message) {
        if (errorMessage) {
            errorMessage.textContent = message;
            errorMessage.style.display = 'block';
            setTimeout(() => {
                errorMessage.style.display = 'none';
            }, 3000);
        }
    }

    // Handle compare button click
    if (compareBtn) {
        compareBtn.addEventListener('click', async function() {
            const formData = new FormData();
            formData.append('fingerprint1', fileInputs[0].files[0]);
            formData.append('fingerprint2', fileInputs[1].files[0]);

            compareBtn.disabled = true;
            compareBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>جاري المعالجة...';
            if (errorMessage) errorMessage.style.display = 'none';
            if (resultsDiv) resultsDiv.style.display = 'none';

            try {
                const response = await fetch('/upload', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                if (response.ok && data.success) {
                    // إظهار وتحديث النتائج
                    if (resultsDiv) {
                        resultsDiv.style.display = 'block';
                        resultsDiv.classList.add('show');
                    }
                    
                    // تحديث نسبة التطابق
                    const score = Math.round(data.match_score * 100);
                    if (progressBar) progressBar.style.width = `${score}%`;
                    if (matchScore) matchScore.textContent = `${score}%`;
                    
                    const pointsCount = document.getElementById('matching-points-count');
                    if (pointsCount) pointsCount.textContent = data.matching_points.length;
                    
                    // تحديث الصور المعالجة
                    if (data.marked1 && previewImages[0]) {
                        previewImages[0].src = data.marked1;
                    }
                    if (data.marked2 && previewImages[1]) {
                        previewImages[1].src = data.marked2;
                    }

                    // عرض صورة خطوط التطابق
                    const matchingImage = document.getElementById('matching-image');
                    if (data.matching_visualization && matchingImage) {
                        matchingImage.src = data.matching_visualization;
                        matchingImage.style.display = 'block';
                    }

                    // تحديث معلومات النقاط المميزة
                    if (matchingPoints) {
                        matchingPoints.innerHTML = `
                            <div class="feature-info">
                                <p>عدد النقاط المميزة في البصمة الأولى: ${data.num_features1}</p>
                                <p>عدد النقاط المميزة في البصمة الثانية: ${data.num_features2}</p>
                                <p>عدد النقاط المتطابقة: ${data.matching_points.length}</p>
                                <div class="match-details mt-3">
                                    <p><span class="text-danger">●</span> نقاط النهاية</p>
                                    <p><span class="text-success">●</span> نقاط التفرع</p>
                                    <p><span class="text-primary">●</span> خطوط التطابق</p>
                                </div>
                            </div>
                        `;
                    }
                } else {
                    throw new Error(data.error || 'فشلت عملية المقارنة');
                }
            } catch (error) {
                showError(error.message);
            } finally {
                if (compareBtn) {
                    compareBtn.disabled = false;
                    compareBtn.innerHTML = '<i class="fas fa-sync-alt me-2"></i>مقارنة البصمات';
                }
            }
        });
    }
});

// Performance Monitoring Class
class PerformanceMonitor {
    constructor() {
        this.startTime = null;
        this.steps = document.querySelectorAll('.step');
        this.progressBar = document.querySelector('.progress-bar');
        this.processingTime = document.getElementById('processing-time');
        this.memoryUsage = document.getElementById('memory-usage');
        this.errorMessage = document.getElementById('error-message');
        
        // Start monitoring system info
        this.startMonitoring();
    }

    startMonitoring() {
        setInterval(() => this.updateSystemInfo(), 1000);
    }

    start() {
        this.startTime = Date.now();
        this.resetSteps();
        this.updateProgress(0);
        this.hideError();
    }

    updateProgress(percent) {
        this.progressBar.style.width = `${percent}%`;
    }

    updateStep(stepIndex, status) {
        const step = this.steps[stepIndex];
        if (!step) return;

        // Remove all status classes
        step.classList.remove('completed', 'in-progress', 'failed');
        
        // Update status icon and add appropriate class
        const statusIcon = step.querySelector('.step-status');
        switch (status) {
            case 'completed':
                statusIcon.textContent = '✅';
                step.classList.add('completed');
                break;
            case 'in-progress':
                statusIcon.textContent = '⏳';
                step.classList.add('in-progress');
                break;
            case 'failed':
                statusIcon.textContent = '❌';
                step.classList.add('failed');
                break;
            default:
                statusIcon.textContent = '⭕';
        }
    }

    showError(message) {
        this.errorMessage.textContent = message;
        this.errorMessage.style.display = 'block';
    }

    hideError() {
        this.errorMessage.style.display = 'none';
    }

    resetSteps() {
        this.steps.forEach(step => {
            step.classList.remove('completed', 'in-progress', 'failed');
            const statusIcon = step.querySelector('.step-status');
            statusIcon.textContent = '⭕';
        });
    }

    updateSystemInfo() {
        if (this.startTime) {
            const elapsedTime = ((Date.now() - this.startTime) / 1000).toFixed(1);
            this.processingTime.textContent = elapsedTime;
        }

        // Get memory usage if available
        if (window.performance && window.performance.memory) {
            const memoryMB = (window.performance.memory.usedJSHeapSize / (1024 * 1024)).toFixed(1);
            this.memoryUsage.textContent = memoryMB;
        }
    }
}

// Initialize performance monitor
const performanceMonitor = new PerformanceMonitor();

// Update the file upload handling
document.getElementById('compare-btn').addEventListener('click', async () => {
    const file1 = document.getElementById('fingerprint1').files[0];
    const file2 = document.getElementById('fingerprint2').files[0];

    if (!file1 || !file2) {
        performanceMonitor.showError('الرجاء اختيار ملفين للبصمات');
        return;
    }

    const formData = new FormData();
    formData.append('fingerprint1', file1);
    formData.append('fingerprint2', file2);

    try {
        performanceMonitor.start();
        performanceMonitor.updateStep(0, 'in-progress');
        performanceMonitor.updateProgress(25);

        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('فشل في معالجة الطلب');
        }

        performanceMonitor.updateStep(0, 'completed');
        performanceMonitor.updateStep(1, 'in-progress');
        performanceMonitor.updateProgress(50);

        const data = await response.json();
        
        performanceMonitor.updateStep(1, 'completed');
        performanceMonitor.updateStep(2, 'in-progress');
        performanceMonitor.updateProgress(75);

        // Update UI with results
        updateResults(data);

        performanceMonitor.updateStep(2, 'completed');
        performanceMonitor.updateStep(3, 'completed');
        performanceMonitor.updateProgress(100);

    } catch (error) {
        performanceMonitor.showError(error.message);
        performanceMonitor.updateStep(3, 'failed');
    }
});

function updateResults(data) {
    // تحديث نسبة التطابق
    const matchScore = document.getElementById('match-score');
    const progressBar = document.querySelector('.progress-bar');
    const score = Math.round(data.match_score * 100);
    
    matchScore.textContent = `${score}%`;
    progressBar.style.width = `${score}%`;
    
    // إضافة تأثير التحديث
    matchScore.classList.add('updated');
    progressBar.classList.add('updated');
    setTimeout(() => {
        matchScore.classList.remove('updated');
        progressBar.classList.remove('updated');
    }, 500);

    // تحديث عدد النقاط المتطابقة
    const pointsCount = document.getElementById('matching-points-count');
    pointsCount.textContent = data.matching_points.length;
    pointsCount.classList.add('updated');
    setTimeout(() => pointsCount.classList.remove('updated'), 500);

    // تحديث صورة المقارنة
    const matchingImage = document.getElementById('matching-image');
    matchingImage.src = data.matching_visualization;
    matchingImage.style.display = 'block';
    matchingImage.classList.add('updated');
    setTimeout(() => matchingImage.classList.remove('updated'), 500);

    // تحديث قائمة النقاط المتطابقة
    const matchingPointsContainer = document.getElementById('matching-points');
    matchingPointsContainer.innerHTML = '';
    
    data.matching_points.forEach((point, index) => {
        const pointElement = document.createElement('div');
        pointElement.className = 'matching-point';
        pointElement.innerHTML = `
            <i class="fas fa-circle text-primary me-2"></i>
            <span>نقطة تطابق ${index + 1}: (${point.x1}, ${point.y1}) ↔ (${point.x2}, ${point.y2})</span>
        `;
        matchingPointsContainer.appendChild(pointElement);
    });
    
    matchingPointsContainer.classList.add('updated');
    setTimeout(() => matchingPointsContainer.classList.remove('updated'), 500);
} 