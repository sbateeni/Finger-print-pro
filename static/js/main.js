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
    const ctx = canvas.getContext('2d');

    // Set canvas size
    function resizeCanvas() {
        const container = canvas.parentElement;
        canvas.width = container.offsetWidth;
        canvas.height = container.offsetHeight;
    }
    
    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();

    // Draw matching points and lines
    function drawMatchingPoints(points, img1, img2) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Calculate scaling factors
        const maxWidth = canvas.width / 2;
        const maxHeight = canvas.height;
        
        const scale1 = Math.min(maxWidth / img1.naturalWidth, maxHeight / img1.naturalHeight);
        const scale2 = Math.min(maxWidth / img2.naturalWidth, maxHeight / img2.naturalHeight);
        
        // Calculate positions
        const img1Width = img1.naturalWidth * scale1;
        const img1Height = img1.naturalHeight * scale1;
        const img2Width = img2.naturalWidth * scale2;
        const img2Height = img2.naturalHeight * scale2;
        
        const img1X = 0;
        const img1Y = (canvas.height - img1Height) / 2;
        const img2X = canvas.width / 2;
        const img2Y = (canvas.height - img2Height) / 2;
        
        // Draw images
        ctx.drawImage(img1, img1X, img1Y, img1Width, img1Height);
        ctx.drawImage(img2, img2X, img2Y, img2Width, img2Height);
        
        // Draw matching points and lines
        points.forEach((pair, index) => {
            const x1 = img1X + pair[0].x * scale1;
            const y1 = img1Y + pair[0].y * scale1;
            const x2 = img2X + pair[1].x * scale2;
            const y2 = img2Y + pair[1].y * scale2;
            
            // Draw line
            ctx.beginPath();
            ctx.moveTo(x1, y1);
            ctx.lineTo(x2, y2);
            ctx.strokeStyle = `hsl(${(index * 360) / points.length}, 70%, 50%)`;
            ctx.lineWidth = 2;
            ctx.stroke();
            
            // Draw points
            ctx.beginPath();
            ctx.arc(x1, y1, 5, 0, Math.PI * 2);
            ctx.arc(x2, y2, 5, 0, Math.PI * 2);
            ctx.fillStyle = `hsl(${(index * 360) / points.length}, 70%, 50%)`;
            ctx.fill();
        });
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
                updateStatus(`تم تحميل الصورة ${index + 1} بنجاح`);
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
        compareBtn.disabled = !bothFilesSelected;
    }

    // Show error message
    function showError(message) {
        errorMessage.textContent = message;
        errorMessage.style.display = 'block';
        setTimeout(() => {
            errorMessage.style.display = 'none';
        }, 3000);
    }

    // Update processing status
    function updateStatus(message) {
        const statusItem = document.createElement('div');
        statusItem.className = 'status-item';
        statusItem.textContent = message;
        processingStatus.appendChild(statusItem);
        processingStatus.scrollTop = processingStatus.scrollHeight;
    }

    // إضافة دالة لتحديث تفاصيل المراحل
    function updateProcessingDetails(message, type = 'info') {
        const stageLogs = document.querySelector('.stage-logs');
        const logEntry = document.createElement('div');
        logEntry.className = `stage-log ${type}`;
        logEntry.textContent = message;
        stageLogs.appendChild(logEntry);
        stageLogs.scrollTop = stageLogs.scrollHeight;
    }

    // تعديل دالة معالجة المقارنة
    compareBtn.addEventListener('click', async function() {
        const formData = new FormData();
        formData.append('fingerprint1', fileInputs[0].files[0]);
        formData.append('fingerprint2', fileInputs[1].files[0]);

        // تهيئة الواجهة
        compareBtn.classList.add('loading');
        compareBtn.disabled = true;
        errorMessage.style.display = 'none';
        document.querySelector('.stage-logs').innerHTML = '';
        resultsDiv.style.display = 'none';

        updateProcessingDetails('بدء عملية المقارنة...', 'info');

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (response.ok && data.success) {
                // تحديث تفاصيل المراحل
                updateProcessingDetails(`تم تحميل الملفات بنجاح`, 'success');
                updateProcessingDetails(`معالجة الصور...`, 'info');
                updateProcessingDetails(`تم استخراج ${data.num_features1} نقطة مميزة من البصمة الأولى`, 'success');
                updateProcessingDetails(`تم استخراج ${data.num_features2} نقطة مميزة من البصمة الثانية`, 'success');
                updateProcessingDetails(`تم العثور على ${data.matching_points.length} نقاط متطابقة`, 'success');
                
                // إظهار وتحديث النتائج
                resultsDiv.style.display = 'block';
                resultsDiv.classList.add('show');
                
                // تحديث نسبة التطابق
                const score = Math.round(data.match_score * 100);
                const progressBar = resultsDiv.querySelector('.progress-bar');
                progressBar.style.width = `${score}%`;
                document.getElementById('match-score').textContent = `${score}%`;
                document.getElementById('matching-points-count').textContent = data.matching_points.length;
                
                // تحديث الصور المعالجة
                if (data.marked1) {
                    previewImages[0].src = data.marked1;
                }
                if (data.marked2) {
                    previewImages[1].src = data.marked2;
                }

                // عرض صورة خطوط التطابق
                const matchingImage = document.getElementById('matching-image');
                if (data.matching_visualization && matchingImage) {
                    matchingImage.src = data.matching_visualization;
                    matchingImage.style.display = 'block';
                }

                // تحديث معلومات النقاط المميزة
                const matchingPointsDiv = document.getElementById('matching-points');
                if (matchingPointsDiv) {
                    matchingPointsDiv.innerHTML = `
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

                updateProcessingDetails(`اكتملت المقارنة بنجاح - نسبة التطابق: ${score}%`, 'success');
            } else {
                throw new Error(data.error || 'فشلت عملية المقارنة');
            }
        } catch (error) {
            updateProcessingDetails(`حدث خطأ: ${error.message}`, 'error');
            showError(error.message);
        } finally {
            compareBtn.classList.remove('loading');
            compareBtn.disabled = false;
        }
    });

    // تحديث معالجة الملفات المرفوعة
    async function handleFileUpload(files) {
        try {
            // بدء مراقبة الأداء
            performanceMonitor.start();
            
            // التحقق من الملفات
            if (!files || files.length !== 2) {
                performanceMonitor.showError('يرجى اختيار ملفين للبصمات');
                return;
            }

            // تحديث حالة المعالجة
            performanceMonitor.update('تحميل الملفات', 'in-progress');
            
            // إنشاء FormData وإضافة الملفات
            const formData = new FormData();
            formData.append('file1', files[0]);
            formData.append('file2', files[1]);

            // إرسال الطلب
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error('فشل في رفع الملفات');
            }

            const data = await response.json();
            
            // تحديث حالة المعالجة
            performanceMonitor.update('تحميل الملفات', 'completed');
            performanceMonitor.update('معالجة الصور', 'in-progress');

            // تحديث الصور المعالجة
            if (data.marked1) {
                document.querySelector('#preview1').src = data.marked1;
            }
            if (data.marked2) {
                document.querySelector('#preview2').src = data.marked2;
            }

            // تحديث حالة المعالجة
            performanceMonitor.update('معالجة الصور', 'completed');
            performanceMonitor.update('استخراج المميزات', 'in-progress');

            // تحديث معلومات المطابقة
            if (data.match_score !== undefined) {
                document.querySelector('#match-score').textContent = `${data.match_score}%`;
            }
            if (data.matching_points !== undefined) {
                document.querySelector('#matching-points').textContent = data.matching_points;
            }

            // تحديث حالة المعالجة
            performanceMonitor.update('استخراج المميزات', 'completed');
            performanceMonitor.update('مقارنة البصمات', 'in-progress');

            // عرض صورة المطابقة
            if (data.matching_visualization) {
                const matchingCanvas = document.querySelector('#matching-canvas');
                const img = new Image();
                img.onload = () => {
                    const ctx = matchingCanvas.getContext('2d');
                    ctx.clearRect(0, 0, matchingCanvas.width, matchingCanvas.height);
                    ctx.drawImage(img, 0, 0, matchingCanvas.width, matchingCanvas.height);
                };
                img.src = data.matching_visualization;
            }

            // تحديث حالة المعالجة النهائية
            performanceMonitor.update('مقارنة البصمات', 'completed');
            performanceMonitor.hideError();

        } catch (error) {
            console.error('Error:', error);
            performanceMonitor.showError(error.message);
            performanceMonitor.update('مقارنة البصمات', 'failed');
        }
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