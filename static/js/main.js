// وظائف معالجة التحميل
function handleFileUpload(input, previewId) {
    const file = input.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            const preview = document.getElementById(previewId);
            preview.src = e.target.result;
            preview.style.display = 'block';
        }
        reader.readAsDataURL(file);
    }
}

// وظائف معالجة النموذج
function validateForm() {
    const fp1 = document.getElementById('fingerprint1');
    const fp2 = document.getElementById('fingerprint2');
    
    if (!fp1.files[0] || !fp2.files[0]) {
        showMessage('الرجاء تحميل كلا البصمتين', 'error');
        return false;
    }
    
    return true;
}

// وظائف عرض الرسائل
function showMessage(message, type) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;
    messageDiv.textContent = message;
    
    const container = document.querySelector('.container');
    container.insertBefore(messageDiv, container.firstChild);
    
    setTimeout(() => {
        messageDiv.remove();
    }, 3000);
}

// وظائف شريط التقدم
function updateProgress(progress) {
    const progressBar = document.querySelector('.progress-bar-fill');
    progressBar.style.width = `${progress}%`;
}

// وظائف معالجة النتائج
function displayResults(results) {
    const resultsDiv = document.querySelector('.results');
    resultsDiv.innerHTML = '';
    
    // عرض نسبة التطابق
    const scoreDiv = document.createElement('div');
    scoreDiv.className = 'score';
    scoreDiv.innerHTML = `
        <h3>نسبة التطابق</h3>
        <p>${(results.score * 100).toFixed(2)}%</p>
    `;
    resultsDiv.appendChild(scoreDiv);
    
    // عرض عدد النقاط المميزة
    const featuresDiv = document.createElement('div');
    featuresDiv.className = 'features';
    featuresDiv.innerHTML = `
        <h3>النقاط المميزة</h3>
        <p>البصمة الأولى: ${results.features1.count}</p>
        <p>البصمة الثانية: ${results.features2.count}</p>
    `;
    resultsDiv.appendChild(featuresDiv);
    
    // عرض النقاط المتطابقة
    const matchesDiv = document.createElement('div');
    matchesDiv.className = 'matches';
    matchesDiv.innerHTML = `
        <h3>النقاط المتطابقة</h3>
        <p>عدد النقاط المتطابقة: ${results.match_count}</p>
    `;
    resultsDiv.appendChild(matchesDiv);
}

// إضافة مستمعي الأحداث
document.addEventListener('DOMContentLoaded', function() {
    // مستمعي أحداث التحميل
    const fp1Input = document.getElementById('fingerprint1');
    const fp2Input = document.getElementById('fingerprint2');
    
    if (fp1Input) {
        fp1Input.addEventListener('change', function() {
            handleFileUpload(this, 'preview1');
        });
    }
    
    if (fp2Input) {
        fp2Input.addEventListener('change', function() {
            handleFileUpload(this, 'preview2');
        });
    }
    
    // مستمع حدث النموذج
    const form = document.querySelector('form');
    if (form) {
        form.addEventListener('submit', function(e) {
            if (!validateForm()) {
                e.preventDefault();
            }
        });
    }
}); 