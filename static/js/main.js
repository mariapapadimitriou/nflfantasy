// NFL Touchdown Predictions - Main JavaScript

document.addEventListener('DOMContentLoaded', function() {
    const loadDataBtn = document.getElementById('load-data-btn');
    const trainModelBtn = document.getElementById('train-model-btn');
    const retrainModelBtn = document.getElementById('retrain-model-btn');
    const predictWeekBtn = document.getElementById('predict-week-btn');
    const exportBtn = document.getElementById('export-btn');
    
    const seasonInput = document.getElementById('season');
    const weekInput = document.getElementById('week');
    const forceReloadCheckbox = document.getElementById('force-reload');
    
    const statusMessage = document.getElementById('status-message');
    const loadingDiv = document.getElementById('loading');
    const tableContainer = document.getElementById('predictions-table-container');
    const tableHead = document.getElementById('table-head');
    const tableBody = document.getElementById('table-body');

    // Helper function to show status message
    function showMessage(message, type = 'success') {
        statusMessage.textContent = message;
        statusMessage.className = `status-message ${type}`;
        statusMessage.style.display = 'flex';
    }

    // Helper function to show/hide loading
    function setLoading(show) {
        loadingDiv.style.display = show ? 'block' : 'none';
        // Disable buttons while loading
        [loadDataBtn, trainModelBtn, retrainModelBtn, predictWeekBtn, exportBtn].forEach(btn => {
            btn.disabled = show;
        });
    }

    // Helper function to get season and week
    function getSeasonAndWeek() {
        const season = parseInt(seasonInput.value);
        const week = parseInt(weekInput.value);
        return { season, week };
    }

    // Helper function to make API calls
    async function apiCall(url, data) {
        try {
            const response = await fetch(url, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': getCookie('csrftoken')
                },
                body: JSON.stringify(data)
            });
            
            // Check if response is JSON
            const contentType = response.headers.get('content-type');
            if (contentType && contentType.includes('application/json')) {
                return await response.json();
            } else {
                // If not JSON, read as text to see what we got
                const text = await response.text();
                console.error('Non-JSON response:', text.substring(0, 200));
                return { 
                    success: false, 
                    message: `Server returned HTML instead of JSON. Status: ${response.status}. Check console for details.` 
                };
            }
        } catch (error) {
            console.error('API call error:', error);
            return { success: false, message: `Network error: ${error.message}` };
        }
    }

    // Helper function to get CSRF token
    function getCookie(name) {
        if (window.csrftoken) {
            return window.csrftoken;
        }
        let cookieValue = null;
        if (document.cookie && document.cookie !== '') {
            const cookies = document.cookie.split(';');
            for (let i = 0; i < cookies.length; i++) {
                const cookie = cookies[i].trim();
                if (cookie.substring(0, name.length + 1) === (name + '=')) {
                    cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                    break;
                }
            }
        }
        return cookieValue;
    }

    // Load Data button handler
    loadDataBtn.addEventListener('click', async function() {
        const { season, week } = getSeasonAndWeek();
        
        if (!season || !week) {
            showMessage('⚠️ Please enter both season and week.', 'warning');
            return;
        }

        setLoading(true);
        showMessage('Loading data...', 'success');

        const result = await apiCall('/api/load-data/', {
            season: season,
            week: week,
            force_reload: forceReloadCheckbox.checked
        });

        setLoading(false);

        if (result.success) {
            showMessage(result.message, 'success');
        } else {
            showMessage(result.message, 'error');
        }
    });

    // Train Model button handler
    trainModelBtn.addEventListener('click', async function() {
        const { season, week } = getSeasonAndWeek();
        
        if (!season || !week) {
            showMessage('⚠️ Please enter both season and week.', 'warning');
            return;
        }

        setLoading(true);
        showMessage('Training model... This may take several minutes.', 'success');

        const result = await apiCall('/api/train-model/', {
            season: season,
            week: week
        });

        setLoading(false);

        if (result.success) {
            showMessage(result.message, 'success');
        } else {
            if (result.model_exists) {
                showMessage('⚠️ ' + result.message + ' Use Retrain Model to overwrite.', 'warning');
            } else {
                showMessage('❌ ' + result.message, 'error');
            }
        }
    });

    // Retrain Model button handler
    retrainModelBtn.addEventListener('click', async function() {
        const { season, week } = getSeasonAndWeek();
        
        if (!season || !week) {
            showMessage('⚠️ Please enter both season and week.', 'warning');
            return;
        }

        if (!confirm('Are you sure you want to overwrite the existing model?')) {
            return;
        }

        setLoading(true);
        showMessage('Retraining model... This may take several minutes.', 'success');

        const result = await apiCall('/api/retrain-model/', {
            season: season,
            week: week
        });

        setLoading(false);

        if (result.success) {
            showMessage(result.message, 'success');
        } else {
            showMessage('❌ ' + result.message, 'error');
        }
    });

    // Predict Week button handler
    predictWeekBtn.addEventListener('click', async function() {
        const { season, week } = getSeasonAndWeek();
        
        if (!season || !week) {
            showMessage('⚠️ Please enter both season and week.', 'warning');
            return;
        }

        setLoading(true);
        showMessage('Generating predictions...', 'success');

        const result = await apiCall('/api/predict-week/', {
            season: season,
            week: week
        });

        setLoading(false);

        if (result.success) {
            showMessage(result.message, 'success');
            displayPredictionsTable(result.data, result.columns);
        } else {
            showMessage('❌ ' + result.message, 'error');
        }
    });

    // Export button handler
    exportBtn.addEventListener('click', async function() {
        const { season, week } = getSeasonAndWeek();
        
        setLoading(true);

        try {
            const response = await fetch('/api/export-predictions/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': getCookie('csrftoken')
                },
                body: JSON.stringify({ season: season, week: week })
            });

            if (response.ok) {
                const blob = await response.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `nfl_touchdown_predictions_s${season}_w${week}.csv`;
                document.body.appendChild(a);
                a.click();
                window.URL.revokeObjectURL(url);
                document.body.removeChild(a);
                showMessage('✅ Data exported successfully!', 'success');
            } else {
                const result = await response.json();
                showMessage('❌ ' + result.message, 'error');
            }
        } catch (error) {
            showMessage('❌ Error exporting data: ' + error.message, 'error');
        }

        setLoading(false);
    });

    // Store predictions data for feature explanations
    let predictionsData = [];

    // Display predictions table
    function displayPredictionsTable(data, columns) {
        // Store data for feature explanations
        predictionsData = data;
        
        // Clear existing table
        tableHead.innerHTML = '';
        tableBody.innerHTML = '';

        // Filter out internal columns from display
        const displayColumns = columns.filter(col => 
            col !== 'feature_explanations' && col !== 'player_id'
        );

        // Create header
        const headerRow = document.createElement('tr');
        displayColumns.forEach(col => {
            const th = document.createElement('th');
            th.textContent = col.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
            headerRow.appendChild(th);
        });
        // Add "Details" column for feature explanations
        const detailsTh = document.createElement('th');
        detailsTh.textContent = 'Details';
        headerRow.appendChild(detailsTh);
        tableHead.appendChild(headerRow);

        // Create rows
        data.forEach((row, index) => {
            const tr = document.createElement('tr');
            tr.classList.add('clickable-row');
            tr.dataset.index = index;
            
            displayColumns.forEach(col => {
                const td = document.createElement('td');
                let value = row[col];
                
                // Format probability
                if (col === 'probability' && typeof value === 'number') {
                    value = (value * 100).toFixed(2) + '%';
                    td.style.fontWeight = 'bold';
                }
                
                // Format NaN/null values
                if (value === null || value === undefined || (typeof value === 'number' && isNaN(value))) {
                    value = '-';
                }
                
                td.textContent = value;
                tr.appendChild(td);
            });
            
            // Add "View Details" button
            const detailsTd = document.createElement('td');
            const detailsBtn = document.createElement('button');
            detailsBtn.className = 'btn-details';
            detailsBtn.textContent = 'View Details';
            detailsBtn.onclick = (e) => {
                e.stopPropagation();
                showFeatureExplanations(row);
            };
            detailsTd.appendChild(detailsBtn);
            tr.appendChild(detailsTd);
            
            // Make row clickable
            tr.onclick = () => showFeatureExplanations(row);
            
            tableBody.appendChild(tr);
        });

        tableContainer.style.display = 'block';
    }

    // Show feature explanations modal
    function showFeatureExplanations(row) {
        const modal = document.getElementById('feature-modal');
        const modalPlayerName = document.getElementById('modal-player-name');
        const modalPlayerInfo = document.getElementById('modal-player-info');
        const modalFeatures = document.getElementById('modal-features');
        
        // Set player name and info
        modalPlayerName.textContent = `${row.player_name || 'Player'} - Feature Contributions`;
        modalPlayerInfo.innerHTML = `
            <div class="player-info-row">
                <span><strong>Team:</strong> ${row.team || '-'}</span>
                <span><strong>Position:</strong> ${row.position || '-'}</span>
                <span><strong>Against:</strong> ${row.against || '-'}</span>
                <span><strong>Probability:</strong> ${row.probability ? (row.probability * 100).toFixed(2) + '%' : '-'}</span>
            </div>
        `;
        
        // Clear previous features
        modalFeatures.innerHTML = '';
        
        // Check if feature explanations exist
        if (!row.feature_explanations || !row.feature_explanations.contributions) {
            modalFeatures.innerHTML = '<p class="no-features">Feature explanations not available for this player.</p>';
            modal.style.display = 'block';
            return;
        }
        
        const contributions = row.feature_explanations.contributions;
        const baseValue = row.feature_explanations.base_value || 0;
        
        // Sort features by absolute contribution (most impactful first)
        const sortedFeatures = Object.entries(contributions)
            .map(([feature, value]) => ({
                feature: feature,
                contribution: value,
                absContribution: Math.abs(value)
            }))
            .sort((a, b) => b.absContribution - a.absContribution);
        
        // Create feature list
        const featuresList = document.createElement('div');
        featuresList.className = 'features-list';
        
        // Add base value explanation
        const baseDiv = document.createElement('div');
        baseDiv.className = 'feature-item base-value';
        baseDiv.innerHTML = `
            <div class="feature-name">Base Probability</div>
            <div class="feature-value">${(baseValue * 100).toFixed(2)}%</div>
            <div class="feature-description">Average touchdown probability across all players</div>
        `;
        featuresList.appendChild(baseDiv);
        
        // Add top contributing features
        sortedFeatures.forEach(({ feature, contribution }) => {
            const featureDiv = document.createElement('div');
            featureDiv.className = `feature-item ${contribution > 0 ? 'positive' : 'negative'}`;
            
            const featureName = feature.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
            const contributionPercent = (contribution * 100).toFixed(2);
            const sign = contribution > 0 ? '+' : '';
            
            featureDiv.innerHTML = `
                <div class="feature-name">${featureName}</div>
                <div class="feature-value ${contribution > 0 ? 'positive' : 'negative'}">
                    ${sign}${contributionPercent}%
                </div>
                <div class="feature-bar">
                    <div class="feature-bar-fill" style="width: ${Math.abs(contribution) * 100}%"></div>
                </div>
            `;
            featuresList.appendChild(featureDiv);
        });
        
        modalFeatures.appendChild(featuresList);
        modal.style.display = 'block';
    }
    
    // Set up close button and outside click handlers using event delegation
    document.addEventListener('click', function(event) {
        const modal = document.getElementById('feature-modal');
        if (!modal) return;
        
        // Close button clicked
        if (event.target.classList.contains('close-modal')) {
            event.preventDefault();
            event.stopPropagation();
            modal.style.display = 'none';
            return;
        }
        
        // Clicked outside modal (on the backdrop)
        if (event.target === modal) {
            modal.style.display = 'none';
        }
    });
    

    // Check for existing model on page load
    async function checkModelExists() {
        const { season, week } = getSeasonAndWeek();
        if (!season || !week) return;

        try {
            const response = await fetch(`/api/check-model/?season=${season}&week=${week}`);
            const result = await response.json();
            if (result.exists) {
                // Optionally show indicator that model exists
            }
        } catch (error) {
            // Silently fail
        }
    }

    // Check model when inputs change
    seasonInput.addEventListener('change', checkModelExists);
    weekInput.addEventListener('change', checkModelExists);
});

