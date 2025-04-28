// متغيرات عامة
let currentStep = 0;
let isProcessing = false;
let performanceMonitor = null;

// تهيئة التطبيق
document.addEventListener('DOMContentLoaded', function() {
    initializeUploadForm();
    initializePerformanceMonitor();
    setupEventListeners();
});

// تهيئة نموذج الرفع
function initializeUploadForm() {
    const uploadForm = document.getElementById('upload-form');
    const fileInput1 = document.getElementById('fingerprint1');
    const fileInput2 = document.getElementById('fingerprint2');
    const uploadButton = document.getElementById('upload-button');
    
    if (!uploadForm || !fileInput1 || !fileInput2 || !uploadButton) {
        console.error('عناصر الرفع غير موجودة');
        return;
    }

    // معالجة اختيار الملفات
    fileInput1.addEventListener('change', handleFileSelect);
    fileInput2.addEventListener('change', handleFileSelect);
    
    // معالجة إرسال النموذج
    uploadForm.addEventListener('submit', handleFormSubmit);
}

// تهيئة مراقب الأداء
function initializePerformanceMonitor() {
    performanceMonitor = {
        startMonitoring: function() {
            console.log('بدء مراقبة الأداء');
        },
        stopMonitoring: function() {
            console.log('إيقاف مراقبة الأداء');
            return {
                totalTime: 0,
                memoryUsage: 0,
                cpuUsage: 0
            };
        },
        addStep: function(step, status) {
            console.log(`إضافة خطوة: ${step}, الحالة: ${status}`);
        },
        updateStep: function(step, status) {
            console.log(`تحديث خطوة: ${step}, الحالة: ${status}`);
        }
    };
}

// إعداد مستمعي الأحداث
function setupEventListeners() {
    // تحديث حالة المعالجة
    setInterval(updateProcessingStatus, 1000);
    
    // تحديث استخدام الموارد
    setInterval(updateResourceUsage, 2000);
}

// معالجة اختيار الملف
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (!file) return;

    // التحقق من نوع الملف
    if (!isValidFileType(file)) {
        showAlert('نوع الملف غير مدعوم. يرجى اختيار صورة بصمة أصبع صالحة.', 'error');
        event.target.value = '';
        return;
    }

    // التحقق من حجم الملف
    if (file.size > 16 * 1024 * 1024) { // 16MB
        showAlert('حجم الملف كبير جداً. الحد الأقصى هو 16MB.', 'error');
        event.target.value = '';
        return;
    }

    // عرض معاينة الصورة
    const reader = new FileReader();
    reader.onload = function(e) {
        const previewId = event.target.id === 'fingerprint1' ? 'preview1' : 'preview2';
        const preview = document.getElementById(previewId);
        if (preview) {
            preview.src = e.target.result;
            preview.style.display = 'block';
        }
    };
    reader.readAsDataURL(file);
}

// معالجة إرسال النموذج
async function handleFormSubmit(event) {
    event.preventDefault();
    
    if (isProcessing) {
        showAlert('جاري معالجة طلب سابق...', 'warning');
        return;
    }

    const formData = new FormData(event.target);
    const fingerprint1 = formData.get('fingerprint1');
    const fingerprint2 = formData.get('fingerprint2');

    if (!fingerprint1 || !fingerprint2) {
        showAlert('يرجى اختيار صورتين للبصمات', 'error');
        return;
    }

    try {
        isProcessing = true;
        updateProcessingStatus('جاري معالجة الصور...');
        
        // بدء مراقبة الأداء
        performanceMonitor.startMonitoring();
        
        // إرسال الطلب
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error('فشل في معالجة الصور');
        }

        const result = await response.json();
        
        if (result.error) {
            throw new Error(result.error);
        }

        // عرض النتائج
        displayResults(result);
        
    } catch (error) {
        showAlert(error.message, 'error');
    } finally {
        isProcessing = false;
        performanceMonitor.stopMonitoring();
        updateProcessingStatus('');
    }
}

// تحديث حالة المعالجة
function updateProcessingStatus(message = '') {
    const statusElement = document.getElementById('processing-status');
    if (statusElement) {
        statusElement.textContent = message;
        statusElement.style.display = message ? 'block' : 'none';
    }
}

// تحديث استخدام الموارد
function updateResourceUsage() {
    if (!isProcessing) return;

    const memoryElement = document.getElementById('memory-usage');
    const cpuElement = document.getElementById('cpu-usage');
    
    if (memoryElement && cpuElement) {
        // هذه قيم وهمية للتوضيح
        memoryElement.textContent = 'الذاكرة: ' + Math.random().toFixed(2) + ' MB';
        cpuElement.textContent = 'المعالج: ' + Math.random().toFixed(2) + ' %';
    }
}

// عرض النتائج
function displayResults(result) {
    const resultsSection = document.getElementById('results-section');
    if (!resultsSection) return;

    // عرض درجة التطابق
    const matchScoreElement = document.getElementById('match-score');
    if (matchScoreElement) {
        matchScoreElement.textContent = (result.match_score * 100).toFixed(2) + '%';
    }

    // عرض عدد النقاط المتطابقة
    const matchingPointsElement = document.getElementById('matching-points');
    if (matchingPointsElement) {
        matchingPointsElement.textContent = result.matching_points.length;
    }

    // عرض الصور المعالجة
    updateImagePreview('processed1', result.processed_image1);
    updateImagePreview('processed2', result.processed_image2);
    updateImagePreview('matching', result.matching_visualization);

    // عرض إحصائيات الأداء
    displayPerformanceStats(result.performance_stats);

    // إظهار قسم النتائج
    resultsSection.style.display = 'block';
}

// تحديث معاينة الصورة
function updateImagePreview(elementId, imageUrl) {
    const preview = document.getElementById(elementId);
    if (preview) {
        preview.src = imageUrl;
        preview.style.display = 'block';
    }
}

// عرض إحصائيات الأداء
function displayPerformanceStats(stats) {
    const statsElement = document.getElementById('performance-stats');
    if (!statsElement) return;

    statsElement.innerHTML = `
        <p>الوقت الإجمالي: ${stats.totalTime.toFixed(2)} ثانية</p>
        <p>استخدام الذاكرة: ${formatBytes(stats.memoryUsage)}</p>
        <p>استخدام المعالج: ${stats.cpuUsage.toFixed(2)}%</p>
    `;
}

// عرض تنبيه
function showAlert(message, type = 'info') {
    const alertElement = document.createElement('div');
    alertElement.className = `alert alert-${type}`;
    alertElement.textContent = message;
    
    const container = document.querySelector('.container');
    if (container) {
        container.insertBefore(alertElement, container.firstChild);
        
        // إزالة التنبيه بعد 5 ثواني
        setTimeout(() => {
            alertElement.remove();
        }, 5000);
    }
}

// التحقق من نوع الملف
function isValidFileType(file) {
    const validTypes = ['image/jpeg', 'image/png', 'image/gif'];
    return validTypes.includes(file.type);
}

// تنسيق حجم الملف
function formatBytes(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('upload-form');
    const uploadButton = document.getElementById('upload-button');
    const processingStatus = document.getElementById('processing-status');
    const resultsSection = document.getElementById('results-section');
    const preview1 = document.getElementById('preview1');
    const preview2 = document.getElementById('preview2');

    // معاينة الصور قبل الرفع
    function previewImage(input, preview) {
        if (input.files && input.files[0]) {
            const reader = new FileReader();
            reader.onload = (e) => {
                preview.src = e.target.result;
                preview.style.display = 'block';
            };
            reader.readAsDataURL(input.files[0]);
        }
    }

    document.getElementById('fingerprint1').addEventListener('change', function() {
        previewImage(this, preview1);
    });

    document.getElementById('fingerprint2').addEventListener('change', function() {
        previewImage(this, preview2);
    });

    // تحديث حالة المعالجة
    async function updateProcessingStatus() {
        try {
            const response = await fetch('/status');
            const data = await response.json();
            
            if (data.error) {
                return;
            }

            let statusHTML = '<h4>حالة المعالجة:</h4>';
            statusHTML += '<div class="progress mb-3">';
            statusHTML += `<div class="progress-bar" role="progressbar" style="width: ${data.progress}%" aria-valuenow="${data.progress}" aria-valuemin="0" aria-valuemax="100">${data.progress}%</div>`;
            statusHTML += '</div>';
            
            if (data.current_step) {
                statusHTML += `<p>الخطوة الحالية: ${data.current_step}</p>`;
            }

            processingStatus.innerHTML = statusHTML;

            if (data.progress < 100) {
                setTimeout(updateProcessingStatus, 1000);
            }
        } catch (error) {
            console.error('خطأ في تحديث الحالة:', error);
        }
    }

    // معالجة رفع الملفات
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const formData = new FormData(form);
        
        // التحقق من اختيار الملفات
        if (!formData.get('fingerprint1') || !formData.get('fingerprint2')) {
            alert('الرجاء اختيار صورتين للبصمات');
            return;
        }

        try {
            // تعطيل زر الرفع وإظهار حالة المعالجة
            uploadButton.disabled = true;
            uploadButton.innerHTML = '<span class="loading"></span> جاري المعالجة...';
            processingStatus.style.display = 'block';
            resultsSection.style.display = 'none';

            // بدء تحديث حالة المعالجة
            updateProcessingStatus();

            // رفع الملفات
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (result.error) {
                throw new Error(result.error);
            }

            // عرض النتائج
            document.getElementById('match-score').textContent = `${(result.match_score * 100).toFixed(2)}%`;
            document.getElementById('matching-points').textContent = result.matching_points.length;

            // عرض الصور المعالجة
            const processed1 = document.getElementById('processed1');
            const processed2 = document.getElementById('processed2');
            const matching = document.getElementById('matching');

            processed1.src = result.features1;
            processed2.src = result.features2;
            matching.src = result.matching_visualization;

            processed1.style.display = 'block';
            processed2.style.display = 'block';
            matching.style.display = 'block';

            // عرض إحصائيات الأداء
            let performanceHTML = '<h5>خطوات المعالجة:</h5><ul class="list-group">';
            for (const step of result.performance_stats.steps) {
                const status = step.status === 'completed' ? 'نجاح' : 'فشل';
                const statusClass = step.status === 'completed' ? 'success' : 'danger';
                performanceHTML += `
                    <li class="list-group-item d-flex justify-content-between align-items-center">
                        ${step.name}
                        <span class="badge bg-${statusClass}">${status}</span>
                    </li>`;
            }
            performanceHTML += '</ul>';
            
            performanceHTML += `
                <div class="mt-3">
                    <p><strong>الوقت الكلي:</strong> ${result.performance_stats.total_time.toFixed(2)} ثانية</p>
                    <p><strong>استخدام الذاكرة:</strong> ${(result.performance_stats.memory_usage / 1024 / 1024).toFixed(2)} ميجابايت</p>
                </div>`;

            document.getElementById('performance-stats').innerHTML = performanceHTML;

            // إظهار قسم النتائج
            resultsSection.style.display = 'block';

        } catch (error) {
            alert(`حدث خطأ: ${error.message}`);
        } finally {
            // إعادة تفعيل زر الرفع
            uploadButton.disabled = false;
            uploadButton.innerHTML = 'مقارنة البصمات';
            processingStatus.style.display = 'none';
        }
    });

    // تحسين تجربة المستخدم عند السحب والإفلات
    const dropZones = document.querySelectorAll('.form-control[type="file"]');
    
    dropZones.forEach(zone => {
        zone.addEventListener('dragover', (e) => {
            e.preventDefault();
            zone.classList.add('dragover');
        });

        zone.addEventListener('dragleave', () => {
            zone.classList.remove('dragover');
        });

        zone.addEventListener('drop', (e) => {
            e.preventDefault();
            zone.classList.remove('dragover');
            
            if (e.dataTransfer.files.length) {
                zone.files = e.dataTransfer.files;
                const event = new Event('change');
                zone.dispatchEvent(event);
            }
        });
    });
}); 