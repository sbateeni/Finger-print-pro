class PerformanceMonitor {
    constructor() {
        this.progressContainer = document.getElementById('progress-container');
        this.stepsList = document.getElementById('steps-list');
        this.systemInfo = document.getElementById('system-info');
        this.estimatedTime = document.getElementById('estimated-time');
        this.currentStep = document.getElementById('current-step');
        this.cpuUsage = document.getElementById('cpu-usage');
        this.memoryUsage = document.getElementById('memory-usage');
        this.updateInterval = null;
    }

    startMonitoring() {
        this.showProgressContainer();
        this.updateInterval = setInterval(() => this.updateStatus(), 1000);
    }

    stopMonitoring() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
        }
        this.hideProgressContainer();
    }

    showProgressContainer() {
        this.progressContainer.style.display = 'block';
    }

    hideProgressContainer() {
        this.progressContainer.style.display = 'none';
    }

    updateStatus() {
        fetch('/status')
            .then(response => response.json())
            .then(data => {
                this.updateStepsList(data.steps);
                this.updateSystemInfo(data);
                this.updateProgress(data);
            })
            .catch(error => console.error('Error fetching status:', error));
    }

    updateStepsList(steps) {
        this.stepsList.innerHTML = '';
        steps.forEach(step => {
            const stepElement = document.createElement('div');
            stepElement.className = `step ${step.status}`;
            stepElement.innerHTML = `
                <span class="step-name">${step.name}</span>
                <span class="step-status">${this.getStatusText(step.status)}</span>
                ${step.duration ? `<span class="step-duration">${step.duration.toFixed(2)}s</span>` : ''}
            `;
            this.stepsList.appendChild(stepElement);
        });
    }

    updateSystemInfo(data) {
        this.cpuUsage.textContent = `${data.cpu_percent}%`;
        this.memoryUsage.textContent = `${data.memory_percent}%`;
        
        if (data.estimated_completion_time) {
            const remainingTime = Math.ceil((data.estimated_completion_time - Date.now() / 1000));
            this.estimatedTime.textContent = this.formatTime(remainingTime);
        }
    }

    updateProgress(data) {
        if (data.current_step) {
            this.currentStep.textContent = data.current_step;
        }
    }

    getStatusText(status) {
        const statusMap = {
            'in_progress': 'جاري التنفيذ...',
            'completed': 'تم الانتهاء',
            'failed': 'فشل'
        };
        return statusMap[status] || status;
    }

    formatTime(seconds) {
        if (seconds < 60) {
            return `${seconds} ثانية`;
        } else if (seconds < 3600) {
            const minutes = Math.floor(seconds / 60);
            return `${minutes} دقيقة`;
        } else {
            const hours = Math.floor(seconds / 3600);
            const minutes = Math.floor((seconds % 3600) / 60);
            return `${hours} ساعة و ${minutes} دقيقة`;
        }
    }
}

// إنشاء كائن المراقب عند تحميل الصفحة
const performanceMonitor = new PerformanceMonitor();

// تحديث حالة المراقبة عند بدء عملية المقارنة
document.getElementById('compare-btn').addEventListener('click', () => {
    performanceMonitor.startMonitoring();
});

// تهيئة متغيرات الأداء
let performanceData = {
    startTime: null,
    steps: [],
    currentStep: 0,
    totalSteps: 5,
    memoryUsage: 0,
    processingTime: 0
};

// تحديث شريط التقدم
function updateProgress(percentage) {
    const progressBar = document.querySelector('.progress');
    const progressText = document.querySelector('.progress-text');
    
    if (progressBar && progressText) {
        progressBar.style.width = `${percentage}%`;
        progressText.textContent = `${percentage}%`;
    }
}

// تحديث حالة الخطوات
function updateStepStatus(stepIndex, status, duration = null) {
    const stepsList = document.querySelector('.steps-list');
    if (!stepsList) return;

    const step = stepsList.children[stepIndex];
    if (!step) return;

    // إزالة جميع الحالات السابقة
    step.classList.remove('completed', 'in-progress', 'failed');
    
    // إضافة الحالة الجديدة
    step.classList.add(status);
    
    // تحديث نص الحالة
    const statusElement = step.querySelector('.step-status');
    if (statusElement) {
        statusElement.textContent = status === 'completed' ? 'مكتمل' :
                                  status === 'in-progress' ? 'قيد التنفيذ' :
                                  status === 'failed' ? 'فشل' : 'في الانتظار';
    }

    // تحديث المدة إذا تم توفيرها
    if (duration && step.querySelector('.step-duration')) {
        step.querySelector('.step-duration').textContent = `${duration}ms`;
    }
}

// تحديث معلومات النظام
function updateSystemInfo() {
    const systemInfo = document.querySelector('.system-info');
    if (!systemInfo) return;

    // تحديث وقت المعالجة
    const processingTimeElement = systemInfo.querySelector('#processing-time');
    if (processingTimeElement) {
        processingTimeElement.textContent = `${performanceData.processingTime}ms`;
    }

    // تحديث استخدام الذاكرة
    const memoryUsageElement = systemInfo.querySelector('#memory-usage');
    if (memoryUsageElement) {
        memoryUsageElement.textContent = `${performanceData.memoryUsage}MB`;
    }
}

// بدء مراقبة الأداء
function startPerformanceMonitoring() {
    performanceData.startTime = Date.now();
    performanceData.steps = [];
    performanceData.currentStep = 0;
    
    // إعادة تعيين شريط التقدم
    updateProgress(0);
    
    // تحديث حالة جميع الخطوات إلى "في الانتظار"
    const steps = document.querySelectorAll('.step');
    steps.forEach((step, index) => {
        updateStepStatus(index, 'waiting');
    });
}

// تحديث الأداء
function updatePerformance(stepName, status, duration = null) {
    const stepIndex = performanceData.steps.findIndex(step => step.name === stepName);
    if (stepIndex === -1) {
        performanceData.steps.push({
            name: stepName,
            status: status,
            duration: duration
        });
    } else {
        performanceData.steps[stepIndex].status = status;
        performanceData.steps[stepIndex].duration = duration;
    }

    // تحديث واجهة المستخدم
    updateStepStatus(performanceData.currentStep, status, duration);
    performanceData.currentStep++;

    // تحديث شريط التقدم
    const progress = (performanceData.currentStep / performanceData.totalSteps) * 100;
    updateProgress(progress);

    // تحديث معلومات النظام
    performanceData.processingTime = Date.now() - performanceData.startTime;
    performanceData.memoryUsage = Math.round(performance.memory?.usedJSHeapSize / (1024 * 1024)) || 0;
    updateSystemInfo();
}

// إظهار رسالة خطأ
function showError(message) {
    const errorElement = document.querySelector('.error-message');
    if (errorElement) {
        errorElement.textContent = message;
        errorElement.style.display = 'block';
    }
}

// إخفاء رسالة الخطأ
function hideError() {
    const errorElement = document.querySelector('.error-message');
    if (errorElement) {
        errorElement.style.display = 'none';
    }
}

// تصدير الوظائف
window.performanceMonitor = {
    start: startPerformanceMonitoring,
    update: updatePerformance,
    showError: showError,
    hideError: hideError
};

// دالة للتحكم في استهلاك المعالج
function checkCPUUsage(cpuUsage) {
    const cpuUsageElement = document.getElementById('cpu-usage');
    const warningElement = document.getElementById('cpu-warning');
    
    // تحديث قيمة استخدام المعالج
    cpuUsageElement.textContent = `${cpuUsage}%`;
    
    // التحقق من تجاوز الحد المسموح به
    if (cpuUsage > 70) {
        // إضافة تنبيه
        if (!warningElement) {
            const warningDiv = document.createElement('div');
            warningDiv.id = 'cpu-warning';
            warningDiv.className = 'alert alert-warning mt-2';
            warningDiv.textContent = 'تحذير: استخدام المعالج مرتفع! سيتم تخفيض سرعة المعالجة تلقائياً.';
            cpuUsageElement.parentNode.appendChild(warningDiv);
        }
        
        // تخفيض سرعة المعالجة
        return true;
    } else {
        // إزالة التنبيه إذا كان موجوداً
        if (warningElement) {
            warningElement.remove();
        }
        return false;
    }
}

// تعديل دالة updatePerformanceInfo
function updatePerformanceInfo() {
    fetch('/status')
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                console.error('Error fetching performance data:', data.error);
                return;
            }

            // التحقق من استخدام المعالج
            const shouldSlowDown = checkCPUUsage(data.cpu_usage);
            
            // تحديث قيم الأداء
            document.getElementById('memory-usage').textContent = `${data.memory_usage}%`;
            document.getElementById('elapsed-time').textContent = `${data.elapsed_time}s`;
            document.getElementById('steps-count').textContent = data.steps_count;
            
            // إذا كان استخدام المعالج مرتفعاً، قم بتخفيض سرعة التحديث
            const updateInterval = shouldSlowDown ? 4000 : 2000;
            clearInterval(window.performanceInterval);
            window.performanceInterval = setInterval(updatePerformanceInfo, updateInterval);
        })
        .catch(error => {
            console.error('Error fetching performance data:', error);
        });
}

// تهيئة متغير التحديث
window.performanceInterval = setInterval(updatePerformanceInfo, 2000);

// تحديث المعلومات عند تحميل الصفحة
document.addEventListener('DOMContentLoaded', updatePerformanceInfo); 