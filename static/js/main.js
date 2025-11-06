// NFL Touchdown Predictions - Main JavaScript

document.addEventListener('DOMContentLoaded', function() {
    const loadDataBtn = document.getElementById('load-data-btn');
    const trainModelBtn = document.getElementById('train-model-btn');
    const retrainModelBtn = document.getElementById('retrain-model-btn');
    const predictWeekBtn = document.getElementById('predict-week-btn');
    const viewFeaturesBtn = document.getElementById('view-features-btn');
    const exportBtn = document.getElementById('export-btn');
    
    // Debug: Check if button exists
    if (!viewFeaturesBtn) {
        console.error('View Features button not found! ID: view-features-btn');
    }
    
    const seasonInput = document.getElementById('season');
    const weekInput = document.getElementById('week');
    const forceReloadCheckbox = document.getElementById('force-reload');
    
    const statusMessage = document.getElementById('status-message');
    const loadingDiv = document.getElementById('loading');
    const tabsContainer = document.querySelector('.tabs-container');
    const predictionsTab = document.getElementById('predictions-tab');
    const featuresTab = document.getElementById('features-tab');
    const tableContainer = document.getElementById('predictions-table-container');
    const featuresTableContainer = document.getElementById('features-table-container');
    const tableHead = document.getElementById('table-head');
    const tableBody = document.getElementById('table-body');
    const featuresTableHead = document.getElementById('features-table-head');
    const featuresTableBody = document.getElementById('features-table-body');
    const featuresInfoText = document.getElementById('features-info-text');

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
        [loadDataBtn, trainModelBtn, retrainModelBtn, predictWeekBtn, viewFeaturesBtn, exportBtn].forEach(btn => {
            if (btn) btn.disabled = show;
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
            // Show tabs if not already visible
            if (tabsContainer) {
                tabsContainer.style.display = 'block';
                // Switch to predictions tab
                document.querySelector('[data-tab="predictions"]').click();
            }
        } else {
            showMessage('❌ ' + result.message, 'error');
        }
    });

    // Tab switching
    document.querySelectorAll('.tab-button').forEach(button => {
        button.addEventListener('click', function() {
            const tabName = this.dataset.tab;
            
            // Update active tab button
            document.querySelectorAll('.tab-button').forEach(btn => btn.classList.remove('active'));
            this.classList.add('active');
            
            // Show/hide tab content
            document.querySelectorAll('.tab-content').forEach(content => {
                content.classList.remove('active');
                content.style.display = 'none';
            });
            
            if (tabName === 'predictions') {
                predictionsTab.classList.add('active');
                predictionsTab.style.display = 'block';
            } else if (tabName === 'features') {
                featuresTab.classList.add('active');
                featuresTab.style.display = 'block';
            }
        });
    });

    // View Features button handler
    if (viewFeaturesBtn) {
        viewFeaturesBtn.addEventListener('click', async function() {
            const { season, week } = getSeasonAndWeek();
            
            if (!season || !week) {
                showMessage('⚠️ Please enter both season and week.', 'warning');
                return;
            }
            
            setLoading(true);
            showMessage('Loading feature data...', 'info');
            
            try {
                const result = await apiCall('/api/feature-data/', { season: season, week: week });
                
                if (result && result.success) {
                    const msg = `✅ Loaded feature data: ${result.row_count} players, ${result.feature_count || result.feature_columns?.length || 0} features`;
                    showMessage(msg, 'success');
                    
                    // Log missing features if any
                    if (result.missing_features && result.missing_features.length > 0) {
                        console.warn('Missing features:', result.missing_features);
                    }
                    
                    // Store feature columns globally for rendering
                    window.currentFeatureColumns = result.feature_columns;
                    displayFeaturesTable(result.data, result.columns, result.feature_columns);
                    
                    // Show tabs and switch to features tab
                    if (tabsContainer) {
                        tabsContainer.style.display = 'block';
                        const featuresTabBtn = document.querySelector('[data-tab="features"]');
                        if (featuresTabBtn) {
                            featuresTabBtn.click();
                        }
                    }
                } else {
                    showMessage('❌ ' + (result?.message || 'Unknown error loading feature data'), 'error');
                }
            } catch (error) {
                console.error('Error loading feature data:', error);
                showMessage('❌ Error loading feature data: ' + error.message, 'error');
            }
            
            setLoading(false);
        });
    } else {
        console.error('View Features button not found!');
    }

    // Store features table data for filtering/sorting
    let featuresTableData = [];
    let featuresTableColumns = [];
    let featuresTableFiltered = [];
    let currentSortColumn = null;
    let currentSortDirection = 'asc';

    // Display features table
    function displayFeaturesTable(data, columns, featureColumns) {
        // Store data for filtering/sorting
        featuresTableData = data;
        featuresTableColumns = columns;
        featuresTableFiltered = [...data];
        
        // Reset sort state
        currentSortColumn = null;
        currentSortDirection = 'asc';
        
        // Clear existing table
        featuresTableHead.innerHTML = '';
        featuresTableBody.innerHTML = '';
        
        // Update info text with more details
        const featureCount = featureColumns ? featureColumns.length : 0;
        featuresInfoText.textContent = `Showing ${data.length} players with ${featureCount} features (X values used by model)`;
        
        // Populate sort dropdown (remove old listeners first)
        const sortSelect = document.getElementById('features-sort');
        const newSortSelect = sortSelect.cloneNode(true);
        sortSelect.parentNode.replaceChild(newSortSelect, sortSelect);
        
        newSortSelect.innerHTML = '<option value="">-- Select Column --</option>';
        columns.forEach((col, idx) => {
            const option = document.createElement('option');
            option.value = idx;
            let colName = col.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
            option.textContent = colName;
            newSortSelect.appendChild(option);
        });
        
        // Create header with sortable columns
        const headerRow = document.createElement('tr');
        columns.forEach((col, idx) => {
            const th = document.createElement('th');
            // Format column name nicely
            let colName = col.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
            th.textContent = colName;
            th.dataset.columnIndex = idx;
            th.dataset.columnName = col;
            
            // Make columns sortable
            th.classList.add('sortable-header');
            th.style.cursor = 'pointer';
            th.title = 'Click to sort';
            
            // Highlight feature columns (model features)
            if (featureColumns && featureColumns.includes(col)) {
                th.classList.add('feature-column');
                th.title = 'Model feature (X value) - Click to sort';
            }
            
            // Add sort indicator
            const sortIndicator = document.createElement('span');
            sortIndicator.className = 'sort-indicator';
            sortIndicator.textContent = ' ↕';
            th.appendChild(sortIndicator);
            
            // Add click handler for sorting
            th.addEventListener('click', () => {
                sortFeaturesTable(idx, col);
            });
            
            headerRow.appendChild(th);
        });
        featuresTableHead.appendChild(headerRow);
        
        // Render the table
        renderFeaturesTable();
        
        // Set up search input (remove old listeners first)
        const searchInput = document.getElementById('features-search');
        const clearSearchBtn = document.getElementById('clear-search');
        const newSearchInput = searchInput.cloneNode(true);
        searchInput.parentNode.replaceChild(newSearchInput, searchInput);
        const newClearBtn = clearSearchBtn.cloneNode(true);
        clearSearchBtn.parentNode.replaceChild(newClearBtn, clearSearchBtn);
        
        newSearchInput.value = '';
        newSearchInput.addEventListener('input', (e) => {
            filterFeaturesTable(e.target.value);
            newClearBtn.style.display = e.target.value ? 'inline-block' : 'none';
        });
        
        newClearBtn.addEventListener('click', () => {
            newSearchInput.value = '';
            filterFeaturesTable('');
            newClearBtn.style.display = 'none';
        });
        
        // Set up sort buttons (remove old listeners first)
        const sortAscBtn = document.getElementById('sort-asc');
        const sortDescBtn = document.getElementById('sort-desc');
        const newSortAsc = sortAscBtn.cloneNode(true);
        const newSortDesc = sortDescBtn.cloneNode(true);
        sortAscBtn.parentNode.replaceChild(newSortAsc, sortAscBtn);
        sortDescBtn.parentNode.replaceChild(newSortDesc, sortDescBtn);
        
        newSortAsc.addEventListener('click', () => {
            if (currentSortColumn !== null) {
                sortFeaturesTable(currentSortColumn, featuresTableColumns[currentSortColumn], 'asc');
            }
        });
        
        newSortDesc.addEventListener('click', () => {
            if (currentSortColumn !== null) {
                sortFeaturesTable(currentSortColumn, featuresTableColumns[currentSortColumn], 'desc');
            }
        });
        
        // Set up dropdown sort
        newSortSelect.addEventListener('change', (e) => {
            const colIdx = parseInt(e.target.value);
            if (!isNaN(colIdx) && colIdx >= 0) {
                sortFeaturesTable(colIdx, featuresTableColumns[colIdx], currentSortDirection);
            } else {
                currentSortColumn = null;
                featuresTableFiltered = [...featuresTableData];
                filterFeaturesTable(newSearchInput.value);
            }
            updateSortButtonStates();
        });
        
        // Update sort button states
        updateSortButtonStates();
        
        featuresTableContainer.style.display = 'block';
    }
    
    // Filter features table
    function filterFeaturesTable(searchTerm) {
        if (!searchTerm || searchTerm.trim() === '') {
            featuresTableFiltered = [...featuresTableData];
        } else {
            const term = searchTerm.toLowerCase().trim();
            featuresTableFiltered = featuresTableData.filter(row => {
                // Search across all columns
                return featuresTableColumns.some(col => {
                    const value = row[col];
                    if (value === null || value === undefined) return false;
                    return String(value).toLowerCase().includes(term);
                });
            });
        }
        
        // Re-apply current sort if any
        if (currentSortColumn !== null) {
            sortFeaturesTable(currentSortColumn, featuresTableColumns[currentSortColumn], currentSortDirection, false);
        } else {
            renderFeaturesTable();
        }
    }
    
    // Sort features table
    function sortFeaturesTable(columnIndex, columnName, direction = null, updateUI = true) {
        if (direction === null) {
            // Toggle direction if same column, otherwise default to asc
            if (currentSortColumn === columnIndex) {
                currentSortDirection = currentSortDirection === 'asc' ? 'desc' : 'asc';
            } else {
                currentSortDirection = 'asc';
            }
        } else {
            currentSortDirection = direction;
        }
        
        currentSortColumn = columnIndex;
        
        // Update sort dropdown
        if (updateUI) {
            const sortSelectEl = document.getElementById('features-sort');
            if (sortSelectEl) {
                sortSelectEl.value = columnIndex;
            }
        }
        
        // Sort the filtered data
        featuresTableFiltered.sort((a, b) => {
            let valA = a[columnName];
            let valB = b[columnName];
            
            // Handle null/undefined
            if (valA === null || valA === undefined) valA = '';
            if (valB === null || valB === undefined) valB = '';
            
            // Convert to comparable values
            if (typeof valA === 'number' && typeof valB === 'number') {
                return currentSortDirection === 'asc' ? valA - valB : valB - valA;
            }
            
            // String comparison
            const strA = String(valA).toLowerCase();
            const strB = String(valB).toLowerCase();
            
            if (currentSortDirection === 'asc') {
                return strA < strB ? -1 : strA > strB ? 1 : 0;
            } else {
                return strA > strB ? -1 : strA < strB ? 1 : 0;
            }
        });
        
        // Update sort indicators in headers
        document.querySelectorAll('#features-table-head th').forEach((th, idx) => {
            const indicator = th.querySelector('.sort-indicator');
            if (indicator) {
                if (idx === columnIndex) {
                    indicator.textContent = currentSortDirection === 'asc' ? ' ↑' : ' ↓';
                    indicator.style.color = '#43dae5';
                } else {
                    indicator.textContent = ' ↕';
                    indicator.style.color = '#999';
                }
            }
        });
        
        renderFeaturesTable();
        updateSortButtonStates();
    }
    
    // Update sort button states
    function updateSortButtonStates() {
        const sortAscBtn = document.getElementById('sort-asc');
        const sortDescBtn = document.getElementById('sort-desc');
        if (sortAscBtn && sortDescBtn) {
            const disabled = currentSortColumn === null;
            sortAscBtn.disabled = disabled;
            sortDescBtn.disabled = disabled;
            sortAscBtn.style.opacity = disabled ? '0.5' : '1';
            sortDescBtn.style.opacity = disabled ? '0.5' : '1';
        }
    }
    
    // Render the features table
    function renderFeaturesTable() {
        featuresTableBody.innerHTML = '';
        
        // Update count
        const countSpan = document.getElementById('features-count');
        if (countSpan) {
            countSpan.textContent = featuresTableFiltered.length;
        }
        
        // Create rows
        featuresTableFiltered.forEach(row => {
            const tr = document.createElement('tr');
            
            featuresTableColumns.forEach(col => {
                const td = document.createElement('td');
                let value = row[col];
                
                // Format NaN/null values
                if (value === null || value === undefined || (typeof value === 'number' && isNaN(value))) {
                    value = '-';
                    td.style.color = '#666';
                } else if (typeof value === 'number') {
                    // Format numbers with appropriate precision
                    if (col.includes('ewma') || col.includes('share') || col.includes('probability')) {
                        value = value.toFixed(4);
                    } else if (col.includes('yards') || col.includes('touchdowns')) {
                        value = value.toFixed(1);
                    } else {
                        value = value.toFixed(2);
                    }
                }
                
                td.textContent = value;
                // Highlight feature columns (model features)
                const featureColumns = featuresTableData.length > 0 ? 
                    (window.currentFeatureColumns || []) : [];
                if (featureColumns && featureColumns.includes(col)) {
                    td.classList.add('feature-cell');
                }
                tr.appendChild(td);
            });
            
            featuresTableBody.appendChild(tr);
        });
    }

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

