document.addEventListener('DOMContentLoaded', function() {
    const uploadForm = document.getElementById('uploadForm');
    const matchForm = document.getElementById('matchForm');
    const uploadResult = document.getElementById('uploadResult');
    const matchResult = document.getElementById('matchResult');
    const resultsContainer = document.getElementById('resultsContainer');

    // Handle fingerprint upload
    uploadForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        const formData = new FormData(this);
        
        try {
            const response = await fetch('/upload', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (response.ok) {
                uploadResult.innerHTML = `
                    <div class="alert alert-success">
                        <h4>Upload Successful</h4>
                        <p>Quality Score: ${data.quality_score}</p>
                        <p>Core Point: (${data.core[0]}, ${data.core[1]})</p>
                        <p>Delta Point: (${data.delta[0]}, ${data.delta[1]})</p>
                    </div>
                `;
                
                // Display the uploaded image
                const file = formData.get('fingerprint');
                const reader = new FileReader();
                reader.onload = function(e) {
                    resultsContainer.innerHTML = `
                        <div class="row">
                            <div class="col-md-6">
                                <h4>Uploaded Fingerprint</h4>
                                <img src="${e.target.result}" class="img-fluid" alt="Uploaded Fingerprint">
                            </div>
                        </div>
                    `;
                };
                reader.readAsDataURL(file);
            } else {
                uploadResult.innerHTML = `
                    <div class="alert alert-danger">
                        <h4>Upload Failed</h4>
                        <p>${data.error}</p>
                    </div>
                `;
            }
        } catch (error) {
            uploadResult.innerHTML = `
                <div class="alert alert-danger">
                    <h4>Error</h4>
                    <p>${error.message}</p>
                </div>
            `;
        }
    });

    // Handle fingerprint matching
    matchForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        const formData = new FormData(this);
        
        try {
            const response = await fetch('/match', {
                method: 'POST',
                body: formData
            });
            
            const data = await response.json();
            
            if (response.ok) {
                matchResult.innerHTML = `
                    <div class="alert alert-success">
                        <h4>Matching Results</h4>
                        <p>Found ${data.matches.length} potential matches</p>
                    </div>
                `;
                
                // Display the matched image and results
                const file = formData.get('fingerprint');
                const reader = new FileReader();
                reader.onload = function(e) {
                    resultsContainer.innerHTML = `
                        <div class="row">
                            <div class="col-md-6">
                                <h4>Input Fingerprint</h4>
                                <img src="${e.target.result}" class="img-fluid" alt="Input Fingerprint">
                            </div>
                            <div class="col-md-6">
                                <h4>Matching Results</h4>
                                <div class="list-group">
                                    ${data.matches.map(match => `
                                        <div class="list-group-item">
                                            <h5>Match Score: ${match.score}</h5>
                                            <p>Algorithm: ${match.algorithm}</p>
                                        </div>
                                    `).join('')}
                                </div>
                            </div>
                        </div>
                    `;
                };
                reader.readAsDataURL(file);
            } else {
                matchResult.innerHTML = `
                    <div class="alert alert-danger">
                        <h4>Matching Failed</h4>
                        <p>${data.error}</p>
                    </div>
                `;
            }
        } catch (error) {
            matchResult.innerHTML = `
                <div class="alert alert-danger">
                    <h4>Error</h4>
                    <p>${error.message}</p>
                </div>
            `;
        }
    });
}); 