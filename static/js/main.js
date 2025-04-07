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

    // Handle comparison
    compareBtn.addEventListener('click', async function() {
        const formData = new FormData();
        formData.append('fingerprint1', fileInputs[0].files[0]);
        formData.append('fingerprint2', fileInputs[1].files[0]);

        // Show loading state
        compareBtn.classList.add('loading');
        compareBtn.disabled = true;
        errorMessage.style.display = 'none';
        resultsDiv.style.display = 'none';
        processingStatus.innerHTML = '';
        updateStatus('بدء عملية المقارنة...');

        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (response.ok && data.success) {
                // Update results
                const score = Math.round(data.match_score * 100);
                progressBar.style.width = `${score}%`;
                matchScore.textContent = `${score}%`;
                
                // Show results div
                resultsDiv.style.display = 'block';
                
                // Update matching points information
                matchingPoints.innerHTML = `
                    <p>عدد النقاط المميزة في البصمة الأولى: ${data.num_features1}</p>
                    <p>عدد النقاط المميزة في البصمة الثانية: ${data.num_features2}</p>
                    <p>عدد النقاط المتطابقة: ${data.matching_points.length}</p>
                `;
                
                // Wait for images to load before drawing
                Promise.all([
                    new Promise(resolve => previewImages[0].onload = resolve),
                    new Promise(resolve => previewImages[1].onload = resolve)
                ]).then(() => {
                    // Draw matching points
                    if (data.matching_points && data.matching_points.length > 0) {
                        drawMatchingPoints(data.matching_points, previewImages[0], previewImages[1]);
                    }
                });
                
                // Update status with feature information
                updateStatus(`تم استخراج ${data.num_features1} نقطة مميزة من البصمة الأولى`);
                updateStatus(`تم استخراج ${data.num_features2} نقطة مميزة من البصمة الثانية`);
                
                // Display matching points
                matchingPoints.innerHTML = '';
                if (data.matching_points && data.matching_points.length > 0) {
                    updateStatus(`تم العثور على ${data.matching_points.length} نقاط متطابقة`);
                    const pointsList = document.createElement('ul');
                    data.matching_points.forEach((pair, index) => {
                        const li = document.createElement('li');
                        li.textContent = `نقطة تطابق ${index + 1}: (${pair[0].x}, ${pair[0].y}) ↔ (${pair[1].x}, ${pair[1].y})`;
                        pointsList.appendChild(li);
                    });
                    matchingPoints.appendChild(pointsList);
                }

                updateStatus(`نسبة التطابق النهائية: ${score}%`);
            } else {
                throw new Error(data.error || 'فشلت عملية المقارنة');
            }
        } catch (error) {
            showError(error.message);
            updateStatus(`حدث خطأ: ${error.message}`);
        } finally {
            compareBtn.classList.remove('loading');
            compareBtn.disabled = false;
        }
    });
}); 