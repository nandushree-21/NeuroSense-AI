/* =========================================================
   NEUROSENSE AI - FRONTEND SCRIPT
========================================================= */

const API_BASE = "http://127.0.0.1:5000";

let selectedFile = null;
let lastAnalysisResult = null;
let lastSignal = [];
let lastDatasetInfo = {};
let patientLocation = null;


/* =========================================================
   INITIALIZATION
========================================================= */

document.addEventListener("DOMContentLoaded", function () {

    console.log("NeuroSense AI frontend loaded.");

    setupNavigation();
    setupFileUpload();
    setupDragDrop();

    checkBackend();

    loadModelComparison();

    showDefaultHospitals();

});


/* =========================================================
   NAVIGATION
========================================================= */

function setupNavigation() {

    const navItems =
        document.querySelectorAll(".nav-item");

    navItems.forEach(function (button) {

        button.addEventListener(
            "click",
            function () {

                navItems.forEach(function (item) {
                    item.classList.remove("active");
                });

                button.classList.add("active");

            }
        );

    });
}


function showSection(sectionId) {

    const sections =
        document.querySelectorAll(".page-section");

    sections.forEach(function (section) {

        section.classList.remove("active");

    });


    const selectedSection =
        document.getElementById(sectionId);

    if (selectedSection) {

        selectedSection.classList.add("active");

        window.scrollTo({
            top: 0,
            behavior: "smooth"
        });
    }


    const navItems =
        document.querySelectorAll(".nav-item");

    navItems.forEach(function (item) {

        item.classList.remove("active");

        const onclick =
            item.getAttribute("onclick");

        if (
            onclick &&
            onclick.includes(`'${sectionId}'`)
        ) {

            item.classList.add("active");
        }

    });
}


/* =========================================================
   BACKEND HEALTH CHECK
========================================================= */

async function checkBackend() {

    const status =
        document.getElementById("backendStatus");

    const dot =
        document.getElementById("backendDot");


    try {

        const response =
            await fetch(`${API_BASE}/health`);


        if (!response.ok) {
            throw new Error("Backend unavailable");
        }


        const data =
            await response.json();


        status.textContent =
            "Backend Online";

        dot.classList.add("online");


        console.log(
            "Backend health:",
            data
        );

    }

    catch (error) {

        status.textContent =
            "Backend Offline";

        dot.classList.remove("online");

        console.error(
            "Backend error:",
            error
        );
    }
}


/* =========================================================
   FILE UPLOAD
========================================================= */

function setupFileUpload() {

    const input =
        document.getElementById("fileInput");


    if (!input) {
        return;
    }


    input.addEventListener(
        "change",
        function (event) {

            const file =
                event.target.files[0];

            if (file) {

                handleSelectedFile(file);
            }

        }
    );
}


/* =========================================================
   DRAG AND DROP
========================================================= */

function setupDragDrop() {

    const uploadBox =
        document.getElementById("uploadBox");


    if (!uploadBox) {
        return;
    }


    uploadBox.addEventListener(
        "dragover",
        function (event) {

            event.preventDefault();

            uploadBox.classList.add(
                "dragover"
            );

        }
    );


    uploadBox.addEventListener(
        "dragleave",
        function () {

            uploadBox.classList.remove(
                "dragover"
            );

        }
    );


    uploadBox.addEventListener(
        "drop",
        function (event) {

            event.preventDefault();

            uploadBox.classList.remove(
                "dragover"
            );


            const file =
                event.dataTransfer.files[0];


            if (file) {

                handleSelectedFile(file);
            }

        }
    );
}


/* =========================================================
   HANDLE SELECTED EEG FILE
========================================================= */

function handleSelectedFile(file) {

    if (!file.name.toLowerCase().endsWith(".csv")) {

        alert(
            "Please upload an EEG CSV file."
        );

        return;
    }


    selectedFile = file;


    const fileName =
        document.getElementById("fileName");


    if (fileName) {

        fileName.textContent =
            `Selected: ${file.name}`;
    }


    const status =
        document.getElementById(
            "analysisStatus"
        );


    if (status) {

        status.textContent =
            "Reading EEG dataset...";
    }


    readEEGFile(file);
}


/* =========================================================
   READ EEG CSV
========================================================= */

function readEEGFile(file) {

    const reader =
        new FileReader();


    reader.onload =
        function (event) {

            try {

                const csvText =
                    event.target.result;


                const parsed =
                    parseCSV(csvText);


                if (!parsed.headers.length) {

                    throw new Error(
                        "CSV file is empty."
                    );
                }


                const eegColumns =
                    findEEGColumns(
                        parsed.headers
                    );


                if (eegColumns.length === 0) {

                    throw new Error(
                        "Could not find X1-X178 EEG columns."
                    );
                }


                if (parsed.rows.length === 0) {

                    throw new Error(
                        "CSV contains no data rows."
                    );
                }


                /*
                 * Use the first EEG record
                 * for waveform visualization.
                 */

                const firstRow =
                    parsed.rows[0];


                const signal =
                    eegColumns.map(
                        function (column) {

                            const value =
                                parseFloat(
                                    firstRow[column]
                                );

                            return Number.isFinite(value)
                                ? value
                                : 0;
                        }
                    );


                lastSignal = signal;


                lastDatasetInfo = {

                    fileName:
                        file.name,

                    rows:
                        parsed.rows.length,

                    channels:
                        eegColumns.length,

                    headers:
                        parsed.headers

                };


                updateDatasetInfo(
                    parsed,
                    signal,
                    eegColumns
                );


                calculateSignalStatistics(
                    signal
                );


                drawEEGWaveform(
                    signal
                );


                const status =
                    document.getElementById(
                        "analysisStatus"
                    );


                if (status) {

                    status.textContent =
                        "EEG dataset loaded successfully. Ready for analysis.";
                }

            }

            catch (error) {

                console.error(error);

                alert(
                    "Unable to read EEG CSV: " +
                    error.message
                );

            }

        };


    reader.onerror =
        function () {

            alert(
                "Could not read the selected file."
            );

        };


    reader.readAsText(file);
}


/* =========================================================
   SIMPLE CSV PARSER
========================================================= */

function parseCSV(text) {

    const lines =
        text
            .trim()
            .split(/\r?\n/);


    if (lines.length === 0) {

        return {
            headers: [],
            rows: []
        };
    }


    const headers =
        parseCSVLine(
            lines[0]
        );


    const rows = [];


    for (
        let i = 1;
        i < lines.length;
        i++
    ) {

        if (!lines[i].trim()) {
            continue;
        }


        const values =
            parseCSVLine(
                lines[i]
            );


        const row = {};


        headers.forEach(
            function (header, index) {

                row[header] =
                    values[index] !== undefined
                        ? values[index]
                        : "";

            }
        );


        rows.push(row);
    }


    return {
        headers: headers,
        rows: rows
    };
}


/* =========================================================
   CSV LINE PARSER
========================================================= */

function parseCSVLine(line) {

    const result = [];

    let current = "";

    let insideQuotes = false;


    for (
        let i = 0;
        i < line.length;
        i++
    ) {

        const char =
            line[i];


        if (char === '"') {

            if (
                insideQuotes &&
                line[i + 1] === '"'
            ) {

                current += '"';

                i++;

            }
            else {

                insideQuotes =
                    !insideQuotes;
            }

        }

        else if (
            char === "," &&
            !insideQuotes
        ) {

            result.push(
                current.trim()
            );

            current = "";

        }

        else {

            current += char;
        }
    }


    result.push(
        current.trim()
    );


    return result;
}


/* =========================================================
   FIND EEG COLUMNS
========================================================= */

function findEEGColumns(headers) {

    return headers
        .filter(function (header) {

            return /^X\d+$/i.test(
                header.trim()
            );

        })
        .sort(function (a, b) {

            const numberA =
                parseInt(
                    a.substring(1)
                );

            const numberB =
                parseInt(
                    b.substring(1)
                );

            return numberA - numberB;
        });
}


/* =========================================================
   UPDATE DATASET INFORMATION
========================================================= */

function updateDatasetInfo(
    parsed,
    signal,
    eegColumns
) {

    setText(
        "datasetFile",
        lastDatasetInfo.fileName
    );


    setText(
        "datasetSamples",
        parsed.rows.length.toLocaleString()
    );


    setText(
        "datasetChannels",
        eegColumns.length
    );


    const mean =
        calculateMean(signal);


    setText(
        "signalMean",
        mean.toFixed(4)
    );


    setText(
        "analysisSamples",
        signal.length
    );
}


/* =========================================================
   DRAW EEG WAVEFORM
========================================================= */

function drawEEGWaveform(signal) {

    const canvas =
        document.getElementById(
            "eegWaveform"
        );


    if (!canvas || !signal.length) {
        return;
    }


    const container =
        canvas.parentElement;


    const width =
        Math.max(
            container.clientWidth - 20,
            800
        );


    const height = 450;


    const dpr =
        window.devicePixelRatio || 1;


    canvas.width =
        width * dpr;


    canvas.height =
        height * dpr;


    canvas.style.width =
        width + "px";


    canvas.style.height =
        height + "px";


    const ctx =
        canvas.getContext("2d");


    ctx.scale(dpr, dpr);


    // Background
    ctx.fillStyle =
        "#06111c";

    ctx.fillRect(
        0,
        0,
        width,
        height
    );


    const padding = 45;


    const graphWidth =
        width - padding * 2;


    const graphHeight =
        height - padding * 2;


    const min =
        Math.min(...signal);


    const max =
        Math.max(...signal);


    let range =
        max - min;


    if (range === 0) {
        range = 1;
    }


    // Grid
    ctx.strokeStyle =
        "rgba(71, 112, 135, 0.18)";

    ctx.lineWidth = 1;


    const horizontalLines = 8;


    for (
        let i = 0;
        i <= horizontalLines;
        i++
    ) {

        const y =
            padding +
            (graphHeight / horizontalLines) *
            i;


        ctx.beginPath();

        ctx.moveTo(
            padding,
            y
        );

        ctx.lineTo(
            width - padding,
            y
        );

        ctx.stroke();
    }


    const verticalLines = 10;


    for (
        let i = 0;
        i <= verticalLines;
        i++
    ) {

        const x =
            padding +
            (graphWidth / verticalLines) *
            i;


        ctx.beginPath();

        ctx.moveTo(
            x,
            padding
        );

        ctx.lineTo(
            x,
            height - padding
        );

        ctx.stroke();
    }


    // Center line
    const centerY =
        padding +
        graphHeight / 2;


    ctx.strokeStyle =
        "rgba(34, 211, 238, 0.35)";

    ctx.beginPath();

    ctx.moveTo(
        padding,
        centerY
    );

    ctx.lineTo(
        width - padding,
        centerY
    );

    ctx.stroke();


    // EEG signal
    ctx.strokeStyle =
        "#22d3ee";

    ctx.lineWidth = 1.5;

    ctx.beginPath();


    signal.forEach(
        function (value, index) {

            const x =
                padding +
                (index /
                    (signal.length - 1)) *
                graphWidth;


            const normalized =
                (value - min) /
                range;


            const y =
                height -
                padding -
                normalized *
                graphHeight;


            if (index === 0) {

                ctx.moveTo(
                    x,
                    y
                );

            }
            else {

                ctx.lineTo(
                    x,
                    y
                );
            }

        }
    );


    ctx.stroke();


    // Labels
    ctx.fillStyle =
        "#94a3b8";

    ctx.font =
        "12px Arial";


    ctx.fillText(
        "Amplitude",
        8,
        20
    );


    ctx.fillText(
        "Sample Number",
        width - 120,
        height - 10
    );


    ctx.fillText(
        `Min: ${min.toFixed(2)}`,
        8,
        height - 30
    );


    ctx.fillText(
        `Max: ${max.toFixed(2)}`,
        8,
        height - 12
    );


    const waveformInfo =
        document.getElementById(
            "waveformInfo"
        );


    if (waveformInfo) {

        waveformInfo.textContent =
            `Displaying the first EEG record from ${lastDatasetInfo.fileName} using ${signal.length} EEG samples (X1-X${signal.length}).`;
    }
}


/* =========================================================
   WINDOW RESIZE WAVEFORM
========================================================= */

window.addEventListener(
    "resize",
    function () {

        if (lastSignal.length) {

            drawEEGWaveform(
                lastSignal
            );
        }

    }
);


/* =========================================================
   SIGNAL STATISTICS
========================================================= */

function calculateSignalStatistics(signal) {

    if (!signal.length) {
        return;
    }


    const min =
        Math.min(...signal);


    const max =
        Math.max(...signal);


    const mean =
        calculateMean(signal);


    const std =
        calculateStd(signal, mean);


    const energy =
        signal.reduce(
            function (sum, value) {

                return sum +
                    value * value;

            },
            0
        );


    const zeroCrossings =
        calculateZeroCrossings(
            signal
        );


    setText(
        "signalMin",
        min.toFixed(4)
    );


    setText(
        "signalMax",
        max.toFixed(4)
    );


    setText(
        "signalMean2",
        mean.toFixed(4)
    );


    setText(
        "signalStd",
        std.toFixed(4)
    );


    setText(
        "signalEnergy",
        energy.toFixed(2)
    );


    setText(
        "zeroCrossings",
        zeroCrossings
    );


    setText(
        "analysisSamples",
        signal.length
    );


    setText(
        "signalQuality",
        "Good"
    );
}


/* =========================================================
   MEAN
========================================================= */

function calculateMean(values) {

    if (!values.length) {
        return 0;
    }


    return values.reduce(
        function (sum, value) {
            return sum + value;
        },
        0
    ) / values.length;
}


/* =========================================================
   STANDARD DEVIATION
========================================================= */

function calculateStd(
    values,
    mean
) {

    if (!values.length) {
        return 0;
    }


    const variance =
        values.reduce(
            function (sum, value) {

                return sum +
                    Math.pow(
                        value - mean,
                        2
                    );

            },
            0
        ) / values.length;


    return Math.sqrt(variance);
}


/* =========================================================
   ZERO CROSSINGS
========================================================= */

function calculateZeroCrossings(values) {

    let count = 0;


    for (
        let i = 1;
        i < values.length;
        i++
    ) {

        if (
            (values[i - 1] < 0 &&
                values[i] >= 0) ||

            (values[i - 1] >= 0 &&
                values[i] < 0)
        ) {

            count++;
        }
    }


    return count;
}


/* =========================================================
   EEG ANALYSIS
========================================================= */

async function analyzeEEG() {

    if (!selectedFile) {

        alert(
            "Please upload an EEG CSV dataset first."
        );

        showSection("analysis");

        return;
    }


    const button =
        document.getElementById(
            "analyzeButton"
        );


    const status =
        document.getElementById(
            "analysisStatus"
        );


    button.disabled = true;

    button.innerHTML =
        `<span class="loading">
            Analyzing EEG...
        </span>`;


    status.textContent =
        "Uploading EEG dataset to NeuroSense AI backend...";


    try {

        const formData =
            new FormData();


        formData.append(
            "file",
            selectedFile
        );


        const response =
            await fetch(
                `${API_BASE}/predict`,
                {
                    method: "POST",
                    body: formData
                }
            );


        if (!response.ok) {

            const errorText =
                await response.text();

            throw new Error(
                `Backend error ${response.status}: ${errorText}`
            );
        }


        const result =
            await response.json();


        console.log(
            "EEG analysis result:",
            result
        );


        if (result.success === false) {

            throw new Error(
                result.message ||
                "EEG analysis failed."
            );
        }


        lastAnalysisResult =
            result;


        displayPrediction(
            result
        );


        displayAdaptiveResult(
            result
        );


        displayXAI(
            result
        );


        updateResultPage(
            result
        );


        status.innerHTML =
            "✅ EEG analysis completed successfully.";


        showSection("results");

    }

    catch (error) {

        console.error(error);

        status.innerHTML =
            `❌ ${error.message}`;

        alert(
            "EEG analysis failed:\n\n" +
            error.message
        );

    }

    finally {

        button.disabled = false;

        button.textContent =
            "🧠 Analyze EEG Dataset";
    }
}


/* =========================================================
   DISPLAY PREDICTION
========================================================= */

function displayPrediction(result) {

    const prediction =
        result.prediction ||
        "Unknown";


    const confidence =
        Number(
            result.confidence || 0
        );


    const seizureProbability =
        Number(
            result.seizure_probability || 0
        );


    setText(
        "prediction",
        prediction
    );


    setText(
        "confidence",
        `${confidence.toFixed(2)}%`
    );


    setText(
        "seizureProbability",
        `${seizureProbability.toFixed(2)}%`
    );


    setText(
        "samplesAnalyzed",
        result.samples_analyzed ||
        0
    );


    const predictionElement =
        document.getElementById(
            "prediction"
        );


    if (predictionElement) {

        predictionElement.style.color =
            prediction.toLowerCase()
                .includes("seizure")
                ? "#fb7185"
                : "#4ade80";
    }
}


/* =========================================================
   ADAPTIVE MULTI MODEL RESULT
========================================================= */

function displayAdaptiveResult(result) {

    const container =
        document.getElementById(
            "adaptiveResult"
        );


    if (!container) {
        return;
    }


    const adaptive =
        result.adaptive_analysis;


    if (
        !adaptive ||
        !adaptive.length
    ) {

        container.textContent =
            "Adaptive ML information is not available.";

        return;
    }


    let html = "";


    adaptive.forEach(
        function (item, index) {

            html += `
                <div class="adaptive-block">

                    <h3>
                        Sample ${index + 1}
                    </h3>

                    <p>
                        <strong>Decision:</strong>
                        ${item.decision || "—"}
                    </p>

                    <p>
                        <strong>Action:</strong>
                        ${item.action || "—"}
                    </p>
            `;


            if (item.uncertainty) {

                html += `
                    <p>
                        <strong>Agreement:</strong>
                        ${item.uncertainty.agreement_level || "—"}
                    </p>

                    <p>
                        <strong>Agreement Score:</strong>
                        ${formatNumber(
                            item.uncertainty.agreement_score
                        )}%
                    </p>

                    <p>
                        <strong>Uncertainty:</strong>
                        ${item.uncertainty.uncertainty_level || "—"}
                    </p>

                    <p>
                        <strong>Mean Seizure Probability:</strong>
                        ${formatNumber(
                            item.uncertainty.mean_probability
                        )}%
                    </p>
                `;
            }


            if (item.model_predictions) {

                html += `
                    <h4>
                        Model Predictions
                    </h4>

                    <ul class="model-prediction-list">
                `;


                Object.entries(
                    item.model_predictions
                ).forEach(
                    function ([model, value]) {

                        html += `
                            <li>
                                <strong>
                                    ${model}
                                </strong>
                                :
                                ${value.prediction || "—"}
                                -
                                ${formatNumber(
                                    value.probability
                                )}%
                            </li>
                        `;

                    }
                );


                html += `
                    </ul>
                `;
            }


            html += `
                </div>
            `;
        }
    );


    if (result.adaptive_status) {

        html += `
            <div class="adaptive-status">
                <strong>
                    Status:
                </strong>
                ${result.adaptive_status}
            </div>
        `;
    }


    container.innerHTML = html;
}


/* =========================================================
   XAI
========================================================= */

function displayXAI(result) {

    const container =
        document.getElementById(
            "xaiContent"
        );


    if (!container) {
        return;
    }


    const xai =
        result.xai;


    if (!xai) {

        container.innerHTML =
            "<p>No explainability data available.</p>";

        return;
    }


    let html =
        `<h3>EEG Signal Explanation</h3>`;


    if (xai.overall) {

        html += `
            <div class="xai-block">

                <h4>
                    Overall Signal
                </h4>

                <p>
                    Mean:
                    ${formatNumber(
                        xai.overall.mean
                    )}
                </p>

                <p>
                    Standard Deviation:
                    ${formatNumber(
                        xai.overall.std
                    )}
                </p>

                <p>
                    Minimum:
                    ${formatNumber(
                        xai.overall.min
                    )}
                </p>

                <p>
                    Maximum:
                    ${formatNumber(
                        xai.overall.max
                    )}
                </p>

                <p>
                    Energy:
                    ${formatNumber(
                        xai.overall.energy
                    )}
                </p>

            </div>
        `;
    }


    if (
        xai.highest_energy_region !==
        undefined
    ) {

        html += `
            <div class="xai-block">

                <h4>
                    Highest Energy Region
                </h4>

                <p>
                    ${xai.highest_energy_region}
                </p>

            </div>
        `;
    }


    if (
        xai.regions &&
        Array.isArray(xai.regions)
    ) {

        html += `
            <div class="xai-block">

                <h4>
                    Signal Regions
                </h4>

                <div class="xai-regions">
        `;


        xai.regions.forEach(
            function (region) {

                html += `
                    <div class="xai-region">

                        <strong>
                            ${region.region ||
                            region.name ||
                            "Region"}
                        </strong>

                        <p>
                            Energy:
                            ${formatNumber(
                                region.energy
                            )}
                        </p>

                        <p>
                            Standard Deviation:
                            ${formatNumber(
                                region.std
                            )}
                        </p>

                    </div>
                `;

            }
        );


        html += `
                </div>
            </div>
        `;
    }


    // Fallback for other backend structures
    if (html ===
        `<h3>EEG Signal Explanation</h3>`) {

        html += `
            <pre class="xai-json">
${escapeHTML(
    JSON.stringify(
        xai,
        null,
        2
    )
)}
            </pre>
        `;
    }


    container.innerHTML = html;
}


/* =========================================================
   RESULT PAGE
========================================================= */

function updateResultPage(result) {

    setText(
        "resultPrediction",
        result.prediction || "—"
    );


    setText(
        "resultConfidence",
        `${formatNumber(
            result.confidence
        )}%`
    );


    setText(
        "resultMessage",
        result.message ||
        "AI screening completed."
    );
}


/* =========================================================
   MODEL COMPARISON
========================================================= */

async function loadModelComparison() {

    try {

        const response =
            await fetch(
                `${API_BASE}/model-comparison`
            );


        if (!response.ok) {
            return;
        }


        const data =
            await response.json();


        console.log(
            "Model comparison:",
            data
        );


        /*
         * The static table in HTML remains
         * as a fallback if the backend response
         * format is different.
         */

        if (
            data &&
            Array.isArray(data.models)
        ) {

            updateModelTable(
                data.models
            );
        }

    }

    catch (error) {

        console.log(
            "Model comparison unavailable:",
            error
        );
    }
}


function updateModelTable(models) {

    const table =
        document.getElementById(
            "modelTable"
        );


    if (!table) {
        return;
    }


    table.innerHTML = "";


    models.forEach(
        function (model) {

            const row =
                document.createElement("tr");


            row.innerHTML = `
                <td>
                    ${model.model ||
                    model.name ||
                    "Model"}
                </td>

                <td>
                    ${formatPercentage(
                        model.accuracy
                    )}
                </td>

                <td>
                    ${formatPercentage(
                        model.precision
                    )}
                </td>

                <td>
                    ${formatPercentage(
                        model.recall
                    )}
                </td>

                <td>
                    ${formatPercentage(
                        model.f1 ||
                        model.f1_score
                    )}
                </td>
            `;


            table.appendChild(row);
        }
    );
}


/* =========================================================
   PATIENT GPS LOCATION
========================================================= */

function getPatientLocation() {

    const status =
        document.getElementById(
            "patientLocationStatus"
        );


    if (!navigator.geolocation) {

        status.textContent =
            "GPS is not supported by this browser.";

        return;
    }


    status.textContent =
        "Detecting location...";


    navigator.geolocation.getCurrentPosition(

        function (position) {

            patientLocation = {

                latitude:
                    position.coords.latitude,

                longitude:
                    position.coords.longitude,

                accuracy:
                    position.coords.accuracy
            };


            status.textContent =
                `Location detected ✓ Accuracy: approximately ${Math.round(
                    patientLocation.accuracy
                )} m`;


            const locationInput =
                document.getElementById(
                    "patientLocation"
                );


            if (locationInput) {

                locationInput.value =
                    "Current GPS Location";
            }

        },

        function (error) {

            console.error(
                "GPS error:",
                error
            );


            status.textContent =
                "Location permission denied or unavailable.";

        },

        {
            enableHighAccuracy: true,

            timeout: 10000,

            maximumAge: 0
        }
    );
}


/* =========================================================
   HOSPITAL DATA
========================================================= */

const hospitals = [

    {
        name: "NIMHANS",

        specialty:
            "Neurology & Epilepsy Care",

        address:
            "Hosur Road, Bengaluru",

        phone:
            "080-26995000",

        lat:
            12.9438,

        lng:
            77.5968,

        website:
            "https://www.nimhans.ac.in/",

        search:
            "NIMHANS Bengaluru"
    },


    {
        name:
            "Ramaiah Memorial Hospital",

        specialty:
            "Neurology & Epilepsy Services",

        address:
            "New BEL Road, Bengaluru",

        phone:
            "080-45366666",

        lat:
            13.0328,

        lng:
            77.5648,

        website:
            "https://www.msrmh.com/",

        search:
            "Ramaiah Memorial Hospital Bengaluru"
    },


    {
        name:
            "Narayana Health City",

        specialty:
            "Neurology & Neurosciences",

        address:
            "Bommasandra, Bengaluru",

        phone:
            "+91 80 6660 6251",

        lat:
            12.8249,

        lng:
            77.6792,

        website:
            "https://www.narayanahealth.org/",

        search:
            "Narayana Health City Bengaluru"
    },


    {
        name:
            "Manipal Hospital Old Airport Road",

        specialty:
            "Neurology & Epilepsy Clinic",

        address:
            "Old Airport Road, Bengaluru",

        phone:
            "1800 102 4647",

        lat:
            12.9588,

        lng:
            77.6470,

        website:
            "https://www.manipalhospitals.com/",

        search:
            "Manipal Hospital Old Airport Road Bengaluru"
    },


    {
        name:
            "Aster RV Hospital",

        specialty:
            "Neurology & Neurosciences",

        address:
            "JP Nagar, Bengaluru",

        phone:
            "080-66040400",

        lat:
            12.9082,

        lng:
            77.5955,

        website:
            "https://www.asterhospitals.in/",

        search:
            "Aster RV Hospital Bengaluru"
    },


    {
        name:
            "Apollo Hospitals Bannerghatta Road",

        specialty:
            "Neurology & Neurosurgery",

        address:
            "Bannerghatta Road, Bengaluru",

        phone:
            "080-40656565",

        lat:
            12.8950,

        lng:
            77.5970,

        website:
            "https://www.apollohospitals.com/",

        search:
            "Apollo Hospitals Bannerghatta Road Bengaluru"
    },


    {
        name:
            "Fortis Hospital Cunningham Road",

        specialty:
            "Neurology & Neurosciences",

        address:
            "Cunningham Road, Bengaluru",

        phone:
            "080-66214444",

        lat:
            12.9975,

        lng:
            77.5948,

        website:
            "https://www.fortishealthcare.com/",

        search:
            "Fortis Hospital Cunningham Road Bengaluru"
    },


    {
        name:
            "Brains Super Speciality Hospital",

        specialty:
            "Neurology & Neurosurgery",

        address:
            "Jayanagar, Bengaluru",

        phone:
            "+91 9483240925",

        lat:
            12.9270,

        lng:
            77.5830,

        website:
            "https://www.brainssuperspecialityhospital.com/",

        search:
            "Brains Super Speciality Hospital Bengaluru"
    }

];


/* =========================================================
   FIND NEARBY HOSPITALS
========================================================= */

function findHospitals() {

    const status =
        document.getElementById(
            "locationStatus"
        );


    const button =
        document.getElementById(
            "locationButton"
        );


    status.innerHTML =
        "📍 Detecting patient's GPS location...";


    button.disabled = true;

    button.innerText =
        "Detecting Location...";


    if (!navigator.geolocation) {

        status.innerHTML =
            "❌ GPS is not supported. Showing Bengaluru hospitals.";

        showDefaultHospitals();

        button.disabled = false;

        button.innerText =
            "📍 Find Hospitals Near Me";

        return;
    }


    navigator.geolocation.getCurrentPosition(

        function (position) {

            const latitude =
                position.coords.latitude;

            const longitude =
                position.coords.longitude;

            const accuracy =
                position.coords.accuracy;


            patientLocation = {

                latitude:
                    latitude,

                longitude:
                    longitude,

                accuracy:
                    accuracy
            };


            status.innerHTML = `
                <div class="location-success">

                    📍 Patient location detected successfully.

                    <br>

                    <span>
                        GPS accuracy:
                        approximately
                        ${Math.round(accuracy)}
                        meters
                    </span>

                </div>
            `;


            showNearbyHospitals(
                latitude,
                longitude
            );


            button.disabled = false;

            button.innerText =
                "📍 Refresh My Location";

        },


        function (error) {

            console.log(
                "Location error:",
                error
            );


            status.innerHTML = `
                <div class="location-warning">

                    ⚠️ Patient GPS location could not
                    be accessed.

                    <br>

                    Showing Bengaluru referral hospitals.

                </div>
            `;


            showDefaultHospitals();


            button.disabled = false;

            button.innerText =
                "📍 Find Hospitals Near Me";
        },


        {
            enableHighAccuracy: true,

            timeout: 10000,

            maximumAge: 0
        }
    );
}


/* =========================================================
   HAVERSINE DISTANCE
========================================================= */

function calculateDistance(
    lat1,
    lon1,
    lat2,
    lon2
) {

    const earthRadius =
        6371;


    const dLat =
        toRadians(
            lat2 - lat1
        );


    const dLon =
        toRadians(
            lon2 - lon1
        );


    const a =
        Math.sin(dLat / 2) *
        Math.sin(dLat / 2) +

        Math.cos(
            toRadians(lat1)
        ) *

        Math.cos(
            toRadians(lat2)
        ) *

        Math.sin(dLon / 2) *
        Math.sin(dLon / 2);


    const c =
        2 *
        Math.atan2(
            Math.sqrt(a),
            Math.sqrt(1 - a)
        );


    return earthRadius * c;
}


function toRadians(degrees) {

    return degrees *
        Math.PI /
        180;
}


/* =========================================================
   SHOW NEARBY HOSPITALS
========================================================= */

function showNearbyHospitals(
    latitude,
    longitude
) {

    const hospitalList =
        document.getElementById(
            "hospitalList"
        );


    const hospitalsWithDistance =
        hospitals.map(
            function (hospital) {

                return {

                    ...hospital,

                    distance:
                        calculateDistance(
                            latitude,
                            longitude,
                            hospital.lat,
                            hospital.lng
                        )
                };

            }
        );


    // Nearest → farthest
    hospitalsWithDistance.sort(
        function (a, b) {

            return a.distance -
                b.distance;
        }
    );


    hospitalList.innerHTML = "";


    hospitalsWithDistance.forEach(
        function (
            hospital,
            index
        ) {

            const card =
                createHospitalCard(
                    hospital,
                    latitude,
                    longitude,
                    index
                );


            hospitalList.appendChild(
                card
            );
        }
    );


    const status =
        document.getElementById(
            "locationStatus"
        );


    status.innerHTML += `
        <div class="sorting-info">

            ✓ Hospitals sorted from nearest
            to farthest using patient's GPS location.

        </div>
    `;
}


/* =========================================================
   CREATE HOSPITAL CARD
========================================================= */

function createHospitalCard(
    hospital,
    patientLatitude,
    patientLongitude,
    index
) {

    const card =
        document.createElement(
            "div"
        );


    card.className =
        "hospital-card";


    if (index === 0) {

        card.classList.add(
            "nearest-hospital"
        );
    }


    const distance =
        formatDistance(
            hospital.distance
        );


    const directionsURL =
        `https://www.google.com/maps/dir/?api=1` +
        `&origin=${patientLatitude},${patientLongitude}` +
        `&destination=${encodeURIComponent(
            hospital.search
        )}` +
        `&travelmode=driving`;


    const callNumber =
        hospital.phone.replace(
            /[^0-9+]/g,
            ""
        );


    card.innerHTML = `

        <div class="hospital-top">

            <div class="hospital-icon">
                🏥
            </div>

            ${
                index === 0
                    ? `
                    <span class="nearest-badge">
                        ⭐ Nearest
                    </span>
                    `
                    : ""
            }

        </div>


        <h3>
            ${escapeHTML(
                hospital.name
            )}
        </h3>


        <p class="hospital-specialty">
            ${escapeHTML(
                hospital.specialty
            )}
        </p>


        <div class="hospital-distance">

            📍 ${distance} away

        </div>


        <p>
            🏠 ${escapeHTML(
                hospital.address
            )}
        </p>


        <p>
            📞 ${escapeHTML(
                hospital.phone
            )}
        </p>


        <div class="hospital-buttons">

            <a
                href="${directionsURL}"
                target="_blank"
                rel="noopener"
                class="hospital-btn">

                🗺️ Directions

            </a>


            <a
                href="tel:${callNumber}"
                class="hospital-btn">

                📞 Call

            </a>


            <a
                href="${hospital.website}"
                target="_blank"
                rel="noopener"
                class="hospital-btn appointment-btn">

                📅 Appointment

            </a>

        </div>
    `;


    return card;
}


/* =========================================================
   FORMAT DISTANCE
========================================================= */

function formatDistance(distance) {

    if (distance < 1) {

        return `${Math.round(
            distance * 1000
        )} m`;
    }


    if (distance < 10) {

        return `${distance.toFixed(
            1
        )} km`;
    }


    return `${Math.round(
        distance
    )} km`;
}


/* =========================================================
   DEFAULT HOSPITALS
========================================================= */

function showDefaultHospitals() {

    const hospitalList =
        document.getElementById(
            "hospitalList"
        );


    hospitalList.innerHTML = "";


    hospitals.forEach(
        function (hospital) {

            const card =
                document.createElement(
                    "div"
                );


            card.className =
                "hospital-card";


            const mapsURL =
                `https://www.google.com/maps/search/?api=1&query=` +
                encodeURIComponent(
                    hospital.search
                );


            const callNumber =
                hospital.phone.replace(
                    /[^0-9+]/g,
                    ""
                );


            card.innerHTML = `

                <div class="hospital-top">

                    <div class="hospital-icon">
                        🏥
                    </div>

                </div>


                <h3>
                    ${escapeHTML(
                        hospital.name
                    )}
                </h3>


                <p class="hospital-specialty">
                    ${escapeHTML(
                        hospital.specialty
                    )}
                </p>


                <p>
                    🏠 ${escapeHTML(
                        hospital.address
                    )}
                </p>


                <p>
                    📞 ${escapeHTML(
                        hospital.phone
                    )}
                </p>


                <div class="hospital-buttons">

                    <a
                        href="${mapsURL}"
                        target="_blank"
                        rel="noopener"
                        class="hospital-btn">

                        🗺️ Open Maps

                    </a>


                    <a
                        href="tel:${callNumber}"
                        class="hospital-btn">

                        📞 Call

                    </a>


                    <a
                        href="${hospital.website}"
                        target="_blank"
                        rel="noopener"
                        class="hospital-btn appointment-btn">

                        📅 Appointment

                    </a>

                </div>
            `;


            hospitalList.appendChild(
                card
            );
        }
    );
}


/* =========================================================
   DOWNLOAD REPORT
========================================================= */

function downloadReport() {

    if (!lastAnalysisResult) {

        alert(
            "Please analyze an EEG dataset first."
        );

        return;
    }


    const patient =
        getPatientDetails();


    const result =
        lastAnalysisResult;


    const report = `

NEUROSENSE AI
AI-Driven EEG Signal Analysis for Automated
Epileptic Seizure Detection

==================================================

PATIENT DETAILS
==================================================

Patient Name:
${patient.name || "Not provided"}

Patient ID:
${patient.id || "Not provided"}

Age:
${patient.age || "Not provided"}

Sex:
${patient.sex || "Not provided"}

Phone:
${patient.phone || "Not provided"}

Location:
${patient.location || "Not provided"}


==================================================

EEG ANALYSIS
==================================================

Dataset:
${lastDatasetInfo.fileName || "Not provided"}

EEG Channels:
${lastDatasetInfo.channels || "—"}

Samples in Dataset:
${lastDatasetInfo.rows || "—"}

Samples Analyzed:
${result.samples_analyzed || "—"}


==================================================

AI SCREENING RESULT
==================================================

Prediction:
${result.prediction || "—"}

Confidence:
${formatNumber(result.confidence)}%

Seizure Probability:
${formatNumber(result.seizure_probability)}%

Quality:
${result.quality_status || "—"}

Adaptive Status:
${result.adaptive_status || "—"}


==================================================

EEG SIGNAL STATISTICS
==================================================

Minimum:
${getElementText("signalMin")}

Maximum:
${getElementText("signalMax")}

Mean:
${getElementText("signalMean2")}

Standard Deviation:
${getElementText("signalStd")}

Energy:
${getElementText("signalEnergy")}

Zero Crossings:
${getElementText("zeroCrossings")}


==================================================

IMPORTANT MEDICAL DISCLAIMER
==================================================

NeuroSense AI is an AI-based EEG screening aid.
It is not a medical diagnosis and should not replace
evaluation by a qualified healthcare professional.

If the patient is experiencing an emergency or active
seizure, seek immediate medical attention.

==================================================
`;


    const blob =
        new Blob(
            [report],
            {
                type:
                    "text/plain;charset=utf-8"
            }
        );


    const url =
        URL.createObjectURL(blob);


    const link =
        document.createElement("a");


    link.href = url;

    link.download =
        "NeuroSenseAI_EEG_Report.txt";


    document.body.appendChild(
        link
    );


    link.click();


    link.remove();


    URL.revokeObjectURL(
        url
    );
}


/* =========================================================
   SHARE RESULT
========================================================= */

async function shareResult() {

    if (!lastAnalysisResult) {

        alert(
            "Please analyze an EEG dataset first."
        );

        return;
    }


    const result =
        lastAnalysisResult;


    const text =
        `NeuroSense AI EEG Screening Result

Prediction:
${result.prediction || "—"}

Confidence:
${formatNumber(result.confidence)}%

Seizure Probability:
${formatNumber(result.seizure_probability)}%

This is an AI-based EEG screening result,
not a medical diagnosis.`;


    if (
        navigator.share
    ) {

        try {

            await navigator.share({

                title:
                    "NeuroSense AI EEG Result",

                text:
                    text

            });

        }

        catch (error) {

            console.log(
                "Share cancelled:",
                error
            );
        }

    }

    else {

        try {

            await navigator.clipboard.writeText(
                text
            );


            alert(
                "Screening result copied to clipboard."
            );

        }

        catch (error) {

            alert(
                text
            );
        }
    }
}


/* =========================================================
   PATIENT DETAILS
========================================================= */

function getPatientDetails() {

    return {

        name:
            getInputValue(
                "patientName"
            ),

        id:
            getInputValue(
                "patientId"
            ),

        age:
            getInputValue(
                "patientAge"
            ),

        sex:
            getInputValue(
                "patientSex"
            ),

        phone:
            getInputValue(
                "patientPhone"
            ),

        location:
            getInputValue(
                "patientLocation"
            )
    };
}


/* =========================================================
   HELPER FUNCTIONS
========================================================= */

function setText(
    elementId,
    value
) {

    const element =
        document.getElementById(
            elementId
        );


    if (element) {

        element.textContent =
            value !== undefined &&
            value !== null
                ? value
                : "—";
    }
}


function getElementText(
    elementId
) {

    const element =
        document.getElementById(
            elementId
        );


    return element
        ? element.textContent
        : "—";
}


function getInputValue(
    elementId
) {

    const element =
        document.getElementById(
            elementId
        );


    return element
        ? element.value.trim()
        : "";
}


function formatNumber(value) {

    const number =
        Number(value);


    if (!Number.isFinite(number)) {

        return "—";
    }


    return number.toFixed(2);
}


function formatPercentage(value) {

    const number =
        Number(value);


    if (!Number.isFinite(number)) {

        return "—";
    }


    /*
     * If backend sends 0.98,
     * convert to 98%.
     *
     * If backend sends 98,
     * keep 98%.
     */

    const percentage =
        number <= 1
            ? number * 100
            : number;


    return `${percentage.toFixed(2)}%`;
}


/* =========================================================
   HTML ESCAPE
========================================================= */

function escapeHTML(value) {

    if (
        value === undefined ||
        value === null
    ) {

        return "";
    }


    return String(value)
        .replace(
            /&/g,
            "&amp;"
        )
        .replace(
            /</g,
            "&lt;"
        )
        .replace(
            />/g,
            "&gt;"
        )
        .replace(
            /"/g,
            "&quot;"
        )
        .replace(
            /'/g,
            "&#039;"
        );
}


/* =========================================================
   MAKE FUNCTIONS AVAILABLE TO HTML
========================================================= */

window.showSection =
    showSection;

window.analyzeEEG =
    analyzeEEG;

window.findHospitals =
    findHospitals;

window.getPatientLocation =
    getPatientLocation;

window.downloadReport =
    downloadReport;

window.shareResult =
    shareResult;