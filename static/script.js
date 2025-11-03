let sessionId = 'user_' + Math.random().toString(36).substr(2, 9);
// Add this at the top of your existing script.js
let currentLanguage = 'english';

function selectLanguage(lang) {
    currentLanguage = lang;
    
    // Update UI
    document.querySelectorAll('.lang-option').forEach(option => {
        option.classList.remove('selected');
    });
    
    document.querySelector(`.lang-option[onclick="selectLanguage('${lang}')"]`).classList.add('selected');
    
    // You can add language-specific functionality here
    if (lang === 'afrikaans') {
        // Update UI texts to Afrikaans
        document.querySelector('h1').textContent = '📰 NAMIBIËSE VALS NUUS OPSPOORDER';
        document.querySelector('.subtitle').textContent = 'Verifieer nuusbewerings in reële tyd met AI & betroubare Namibiese nuusbronne';
        document.getElementById('claimInput').placeholder = '✍️ Voer \'n nuusbewering in: bv. Namibia wen AFCON 2025';
        document.getElementById('analyzeBtn').textContent = '🔍 Verifieer Bewering';
        document.querySelector('.history-header h3').textContent = '📖 Beweringsgeskiedenis (hierdie sessie)';
        document.querySelector('.history-header label').innerHTML = '<input type="checkbox" id="showHistory" checked onchange="toggleHistory()"> Wys Beweringsgeskiedenis';
    } else {
        // Update UI texts to English
        document.querySelector('h1').textContent = '📰 NAMIBIAN FAKE NEWS DETECTOR';
        document.querySelector('.subtitle').textContent = 'Verify news claims in real-time using AI & trusted Namibian news sources';
        document.getElementById('claimInput').placeholder = '✍️ Enter a news claim: e.g. Namibia wins AFCON 2025';
        document.getElementById('analyzeBtn').textContent = '🔍 Verify Claim';
        document.querySelector('.history-header h3').textContent = '📖 Claim History (this session)';
        document.querySelector('.history-header label').innerHTML = '<input type="checkbox" id="showHistory" checked onchange="toggleHistory()"> Show Claim History';
    }
}


function analyzeClaim() {
    const claim = document.getElementById('claimInput').value.trim();
    const analyzeBtn = document.getElementById('analyzeBtn');
    
    if (!claim) {
        alert('⚠️ Please enter a claim first.');
        return;
    }
    
    // Disable button and show loading
    analyzeBtn.disabled = true;
    analyzeBtn.textContent = '⏳ Analyzing...';
    
    // Show loading state
    showLoading();
    
    fetch('/analyze', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            claim: claim,
            session_id: sessionId
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            throw new Error(data.error);
        }
        displayResults(data);
        updateHistory(data.history);
    })
    .catch(error => {
        console.error('Error:', error);
        showError('Analysis failed: ' + error.message);
    })
    .finally(() => {
        analyzeBtn.disabled = false;
        analyzeBtn.textContent = '🔍 Verify Claim';
    });
}

function showLoading() {
    const resultsSection = document.getElementById('resultsSection');
    resultsSection.style.display = 'block';
    resultsSection.innerHTML = `
        <div class="loading-message">
            <p>🔍 Analyzing claim with AI...</p>
            <div class="spinner"></div>
        </div>
    `;
}

function showError(message) {
    const resultsSection = document.getElementById('resultsSection');
    resultsSection.style.display = 'block';
    resultsSection.innerHTML = `
        <div class="error-message">
            ❌ ${message}
        </div>
    `;
}

function displayResults(data) {
    const resultsSection = document.getElementById('resultsSection');
    
    // Build the entire results section HTML
    resultsSection.innerHTML = `
        <div class="success-message">Analysis complete ✅</div>
        
        <div class="results-grid">
            <div class="verdict-box">
                <h3>Verdict</h3>
                <div class="${data.verdict.toLowerCase() === 'real' ? 'verdict-real' : 'verdict-fake'}">
                    ${data.verdict.toLowerCase() === 'real' ? '🟢 REAL' : '🔴 FAKE'}
                </div>
            </div>
            
            <div class="confidence-box">
                <h3>Confidence</h3>
                <div class="progress-container">
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: ${data.confidence * 100}%"></div>
                    </div>
                    <span class="confidence-text">${data.confidence.toFixed(2)}</span>
                </div>
                ${data.similarity !== null && data.similarity !== undefined ? 
                    `<div style="font-weight:bold;color:${getSimilarityColor(data.similarity)};margin-top:10px;">
                        Evidence similarity: ${data.similarity.toFixed(2)}
                    </div>` : ''
                }
            </div>
        </div>

        <div class="evidence-section">
            <h3>📚 Supporting Evidence</h3>
            <div class="evidence-list">
                ${data.evidence && data.evidence.length > 0 ? 
                    data.evidence.map((ev, idx) => {
                        let title = ev;
                        let url = '#';
                        
                        if (ev.includes('(') && ev.endsWith(')')) {
                            const lastParen = ev.lastIndexOf('(');
                            title = ev.substring(0, lastParen).trim();
                            url = ev.substring(lastParen + 1, ev.length - 1);
                        }
                        
                        return `
                            <div class="evidence-item">
                                <div class="evidence-title">${idx + 1}. ${title}</div>
                                <a href="${url}" target="_blank" class="evidence-link">Read more</a>
                            </div>
                        `;
                    }).join('') : 
                    '<p>No supporting evidence retrieved. Verdict is based only on AI model.</p>'
                }
            </div>
        </div>
    `;
}

function getSimilarityColor(similarity) {
    if (similarity >= 0.55) return '#198754';   // green
    if (similarity >= 0.30) return '#f0ad4e';   // orange
    return '#d9534f';   // red
}

function updateHistory(history) {
    const historyTable = document.getElementById('historyTable');
    if (!history || history.length === 0) {
        historyTable.innerHTML = '<p>No history yet.</p>';
        return;
    }
    
    const tableHTML = `
        <table class="history-table">
            <thead>
                <tr>
                    <th>Claim</th>
                    <th>Verdict</th>
                    <th>Confidence</th>
                </tr>
            </thead>
            <tbody>
                ${history.slice().reverse().map(entry => `
                    <tr>
                        <td>${entry.claim.length > 50 ? entry.claim.substring(0, 50) + '...' : entry.claim}</td>
                        <td style="color: ${entry.verdict.toLowerCase() === 'real' ? 'green' : 'red'}; font-weight: bold;">
                            ${entry.verdict}
                        </td>
                        <td>${entry.confidence.toFixed(2)}</td>
                    </tr>
                `).join('')}
            </tbody>
        </table>
    `;
    
    historyTable.innerHTML = tableHTML;
}

function toggleHistory() {
    const historySection = document.querySelector('.history-section');
    const showHistory = document.getElementById('showHistory').checked;
    historySection.style.display = showHistory ? 'block' : 'none';
}

// Load history on page load
document.addEventListener('DOMContentLoaded', function() {
    fetch(`/history/${sessionId}`)
        .then(response => response.json())
        .then(history => updateHistory(history))
        .catch(error => console.error('Error loading history:', error));
});