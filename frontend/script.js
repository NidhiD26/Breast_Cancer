document.addEventListener('DOMContentLoaded', function() {
    fetchAnalysisData();
    fetchSampleData();
    setupPredictionForm();
});

let samples = [];
let modelStructure = [];

async function fetchAnalysisData() {
    try {
        const response = await fetch('http://127.0.0.1:8000/analysis');
        if (!response.ok) throw new Error('Failed to fetch analysis data.');
        const data = await response.json();
        
        modelStructure = data.model_structure;
        renderAnnStructure(modelStructure); // Initial render
        renderEvaluationMetrics(data.confusion_matrix, data.classification_report);
        renderDatasetInfo(data.dataset_info);

    } catch (error) {
        console.error('Error fetching analysis data:', error);
    }
}

async function fetchSampleData() {
    try {
        const response = await fetch('http://127.0.0.1:8000/samples?num_samples=10');
        if (!response.ok) throw new Error('Failed to fetch sample data.');
        samples = await response.json();
        
        const selector = document.getElementById('sampleSelector');
        selector.innerHTML = samples.map((sample, index) => 
            `<option value="${index}">Sample ${index + 1} (Actual: ${sample.actual_label})</option>`
        ).join('');

    } catch (error) {
        console.error('Error fetching sample data:', error);
    }
}

function loadSample() {
    const selector = document.getElementById('sampleSelector');
    const selectedIndex = selector.value;
    if (selectedIndex < 0 || selectedIndex >= samples.length) return;

    const sampleFeatures = samples[selectedIndex].features;
    for (const key in sampleFeatures) {
        const input = document.getElementById(key);
        if (input) {
            input.value = sampleFeatures[key];
        }
    }
}

function renderAnnStructure(structure, activations = []) {
    const ctx = document.getElementById('annCanvas').getContext('2d');
    if (!ctx || !structure || structure.length === 0) return;

    const maxNeurons = Math.max(...structure);
    const layerGap = 100;
    const neuronRadius = 10;
    const neuronGap = 30;

    ctx.canvas.width = (structure.length - 1) * layerGap + 100;
    ctx.canvas.height = maxNeurons * neuronGap + 50;
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);

    const layerCoords = [];

    // Normalize activations for color mapping
    const maxActivation = activations.length > 0 ? Math.max(...activations) : 1;

    structure.forEach((numNeurons, layerIndex) => {
        const x = 50 + layerIndex * layerGap;
        const layerYCoords = [];
        const startY = (ctx.canvas.height - (numNeurons * neuronGap)) / 2;
        for (let i = 0; i < numNeurons; i++) {
            const y = startY + i * neuronGap;
            layerYCoords.push({ x, y });
            
            ctx.beginPath();
            ctx.arc(x, y, neuronRadius, 0, 2 * Math.PI);

            let color = '#007bff'; // Default color
            if (layerIndex === 1 && activations.length > 0) { // Hidden layer
                const activation = activations[i] / (maxActivation + 1e-5); // Normalize
                const blue = 150 + Math.floor(105 * activation);
                color = `rgb(0, 123, ${blue})`;
            }
            ctx.fillStyle = color;
            ctx.fill();
        }
        layerCoords.push(layerYCoords);
    });

    for (let i = 0; i < layerCoords.length - 1; i++) {
        for (const startNeuron of layerCoords[i]) {
            for (const endNeuron of layerCoords[i + 1]) {
                ctx.beginPath();
                ctx.moveTo(startNeuron.x, startNeuron.y);
                ctx.lineTo(endNeuron.x, endNeuron.y);
                ctx.strokeStyle = 'rgba(0, 0, 0, 0.1)';
                ctx.stroke();
            }
        }
    }
}

function renderEvaluationMetrics(confusionMatrix, classReport) {
    const cmTable = document.getElementById('confusionMatrix');
    cmTable.innerHTML = `
        <thead><tr><th></th><th>Predicted Benign</th><th>Predicted Malignant</th></tr></thead>
        <tbody>
            <tr><th>Actual Benign</th><td>${confusionMatrix[0][0]}</td><td>${confusionMatrix[0][1]}</td></tr>
            <tr><th>Actual Malignant</th><td>${confusionMatrix[1][0]}</td><td>${confusionMatrix[1][1]}</td></tr>
        </tbody>`;

    const crTable = document.getElementById('classificationReport');
    crTable.innerHTML = `
        <thead><tr><th>Metric</th><th>Precision</th><th>Recall</th><th>F1-Score</th><th>Support</th></tr></thead>
        <tbody>
            <tr>
                <td>Benign (0)</td>
                <td>${classReport['0'].precision.toFixed(2)}</td>
                <td>${classReport['0'].recall.toFixed(2)}</td>
                <td>${classReport['0']['f1-score'].toFixed(2)}</td>
                <td>${classReport['0'].support}</td>
            </tr>
            <tr>
                <td>Malignant (1)</td>
                <td>${classReport['1'].precision.toFixed(2)}</td>
                <td>${classReport['1'].recall.toFixed(2)}</td>
                <td>${classReport['1']['f1-score'].toFixed(2)}</td>
                <td>${classReport['1'].support}</td>
            </tr>
        </tbody>`;
}

function renderDatasetInfo(info) {
    const infoDiv = document.getElementById('datasetInfo');
    infoDiv.innerHTML = `
        <p><strong>Total Samples:</strong> ${info.total_samples}</p>
        <p><strong>Training Samples:</strong> ${info.training_samples}</p>
        <p><strong>Testing Samples:</strong> ${info.testing_samples}</p>
        <p><strong>Features:</strong> ${info.feature_names.join(', ')}</p>
        <p><strong>Class Distribution:</strong> Benign (0): ${info.class_distribution['0']}, Malignant (1): ${info.class_distribution['1']}</p>`;
}

function setupPredictionForm() {
    document.getElementById('loadSampleBtn').addEventListener('click', loadSample);

    document.getElementById('predictionForm').addEventListener('submit', async function(event) {
        event.preventDefault();

        const data = {
            mean_radius: parseFloat(document.getElementById('mean_radius').value),
            mean_texture: parseFloat(document.getElementById('mean_texture').value),
            mean_perimeter: parseFloat(document.getElementById('mean_perimeter').value),
            mean_area: parseFloat(document.getElementById('mean_area').value),
            mean_smoothness: parseFloat(document.getElementById('mean_smoothness').value)
        };

        const resultDiv = document.getElementById('result');
        resultDiv.innerHTML = '<p>Predicting...</p>';

        try {
            const response = await fetch('http://127.0.0.1:8000/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Prediction failed.');
            }

            const result = await response.json();
            
            // Update ANN visualization with activations
            renderAnnStructure(modelStructure, result.hidden_layer_activations);

            resultDiv.innerHTML = `
                <p><strong>Prediction:</strong> ${result.prediction_label}</p>
                <p><strong>Probability (Benign):</strong> ${(result.prediction_probabilities.Benign * 100).toFixed(2)}%</p>
                <p><strong>Probability (Malignant):</strong> ${(result.prediction_probabilities.Malignant * 100).toFixed(2)}%</p>`;
        } catch (error) {
            resultDiv.innerHTML = `<p style="color: red;">Error: ${error.message}</p>`;
        }
    });
}