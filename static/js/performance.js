// فئة مراقبة الأداء
class PerformanceMonitor {
    constructor() {
        this.startTime = null;
        this.endTime = null;
        this.steps = [];
        this.errors = [];
        this.warnings = [];
        this.resourceUsage = {
            cpu: [],
            memory: [],
            disk_io: [],
            network_io: []
        };
        this.monitoringInterval = null;
    }

    // بدء المراقبة
    startMonitoring() {
        this.startTime = Date.now();
        this.monitoringInterval = setInterval(() => this.updateResourceUsage(), 1000);
        this.updateUI('start');
    }

    // إيقاف المراقبة
    stopMonitoring() {
        this.endTime = Date.now();
        clearInterval(this.monitoringInterval);
        this.updateUI('stop');
        return this.getStats();
    }

    // إضافة خطوة
    addStep(name, duration = 0, status = 'pending') {
        this.steps.push({
            name,
            duration,
            status,
            timestamp: Date.now()
        });
        this.updateUI('step', { name, status });
    }

    // تحديث خطوة
    updateStep(name, status, duration = null) {
        const step = this.steps.find(s => s.name === name);
        if (step) {
            step.status = status;
            if (duration !== null) {
                step.duration = duration;
            }
            this.updateUI('step', { name, status });
        }
    }

    // تسجيل خطأ
    logError(message, step = null) {
        this.errors.push({
            message,
            step,
            timestamp: Date.now()
        });
        this.updateUI('error', { message, step });
    }

    // تسجيل تحذير
    logWarning(message, step = null) {
        this.warnings.push({
            message,
            step,
            timestamp: Date.now()
        });
        this.updateUI('warning', { message, step });
    }

    // تحديث استخدام الموارد
    updateResourceUsage() {
        if (window.performance && window.performance.memory) {
            const memory = window.performance.memory.usedJSHeapSize / (1024 * 1024);
            this.resourceUsage.memory.push(memory);
        }

        // تحديث واجهة المستخدم
        this.updateUI('resources', this.resourceUsage);
    }

    // الحصول على الإحصائيات
    getStats() {
        const duration = (this.endTime - this.startTime) / 1000;
        
        return {
            total_duration: duration,
            steps_count: this.steps.length,
            errors_count: this.errors.length,
            warnings_count: this.warnings.length,
            resource_usage: {
                cpu: {
                    mean: this.calculateMean(this.resourceUsage.cpu),
                    max: Math.max(...this.resourceUsage.cpu),
                    min: Math.min(...this.resourceUsage.cpu)
                },
                memory: {
                    mean: this.calculateMean(this.resourceUsage.memory),
                    max: Math.max(...this.resourceUsage.memory),
                    min: Math.min(...this.resourceUsage.memory)
                },
                disk_io: {
                    total: this.resourceUsage.disk_io.reduce((a, b) => a + b, 0),
                    mean: this.calculateMean(this.resourceUsage.disk_io)
                },
                network_io: {
                    total: this.resourceUsage.network_io.reduce((a, b) => a + b, 0),
                    mean: this.calculateMean(this.resourceUsage.network_io)
                }
            }
        };
    }

    // حساب المتوسط
    calculateMean(array) {
        if (array.length === 0) return 0;
        return array.reduce((a, b) => a + b, 0) / array.length;
    }

    // تحديث واجهة المستخدم
    updateUI(type, data) {
        switch (type) {
            case 'start':
                document.getElementById('processing-status').textContent = 'جاري المعالجة...';
                break;
                
            case 'stop':
                document.getElementById('processing-status').textContent = 'تم الانتهاء';
                break;
                
            case 'step':
                this.updateStepUI(data.name, data.status);
                break;
                
            case 'error':
                this.showNotification(data.message, 'error');
                break;
                
            case 'warning':
                this.showNotification(data.message, 'warning');
                break;
                
            case 'resources':
                this.updateResourceUI(data);
                break;
        }
    }

    // تحديث واجهة الخطوة
    updateStepUI(name, status) {
        const stepElement = document.querySelector(`.step[data-name="${name}"]`);
        if (stepElement) {
            stepElement.className = `step ${status}`;
            stepElement.querySelector('.step-status').textContent = this.getStatusIcon(status);
        }
    }

    // الحصول على أيقونة الحالة
    getStatusIcon(status) {
        switch (status) {
            case 'completed':
                return '✅';
            case 'in-progress':
                return '⏳';
            case 'failed':
                return '❌';
            default:
                return '⭕';
        }
    }

    // عرض إشعار
    showNotification(message, type) {
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.remove();
        }, 3000);
    }

    // تحديث واجهة الموارد
    updateResourceUI(usage) {
        const metrics = {
            cpu: document.getElementById('cpu-usage'),
            memory: document.getElementById('memory-usage'),
            disk: document.getElementById('disk-usage'),
            network: document.getElementById('network-usage')
        };

        if (usage.cpu.length > 0) {
            metrics.cpu.textContent = `${usage.cpu[usage.cpu.length - 1].toFixed(1)}%`;
        }
        
        if (usage.memory.length > 0) {
            metrics.memory.textContent = `${usage.memory[usage.memory.length - 1].toFixed(1)} MB`;
        }
        
        if (usage.disk_io.length > 0) {
            metrics.disk.textContent = this.formatBytes(usage.disk_io[usage.disk_io.length - 1]);
        }
        
        if (usage.network_io.length > 0) {
            metrics.network.textContent = this.formatBytes(usage.network_io[usage.network_io.length - 1]);
        }
    }

    // تنسيق البايت
    formatBytes(bytes) {
        if (bytes === 0) return '0 B';
        
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        
        return `${(bytes / Math.pow(k, i)).toFixed(2)} ${sizes[i]}`;
    }
}

// تصدير الفئة
window.PerformanceMonitor = PerformanceMonitor;

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