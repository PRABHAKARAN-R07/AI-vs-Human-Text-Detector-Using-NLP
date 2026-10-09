const textInput = document.getElementById("textInput");
const wordCount = document.getElementById("wordCount");
const charCount = document.getElementById("charCount");

const analyzeBtn = document.getElementById("analyzeBtn");
const clearBtn = document.getElementById("clearBtn");

const status = document.getElementById("status");
const result = document.getElementById("result");

const predictionText = document.getElementById("predictionText");
const confidenceText = document.getElementById("confidenceText");
const resultStats = document.getElementById("resultStats");


// Update word and character counts
function updateCounts() {
    const text = textInput.value.trim();

    const words = text ? text.split(/\s+/).length : 0;

    wordCount.textContent = words;
    charCount.textContent = textInput.value.length;
}


// Clear the text and result
function clearText() {
    textInput.value = "";

    updateCounts();

    result.hidden = true;
    result.className = "result";

    status.textContent = "Your prediction will appear here.";

    analyzeBtn.disabled = false;
    analyzeBtn.textContent = "Detect Text";
}


// Send text to Flask and get the prediction
async function analyzeText() {
    const text = textInput.value.trim();

    if (!text) {
        status.textContent = "Please enter a paragraph first.";
        result.hidden = true;
        textInput.focus();
        return;
    }

    analyzeBtn.disabled = true;
    analyzeBtn.textContent = "Analyzing...";
    status.textContent = "Analyzing your text...";

    result.hidden = true;

    try {
        // Connect to Flask API
        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({
                text: text
            })
        });

        // Read the API response
        const data = await response.json();

        if (!response.ok) {
            throw new Error(
                data.error || "The prediction request failed."
            );
        }

        // Read the prediction returned by Python
        const prediction = String(
            data.prediction || ""
        ).trim().toLowerCase();

        if (prediction !== "ai" && prediction !== "human") {
            throw new Error(
                "Unexpected prediction. Check your Flask API response."
            );
        }

        // Display prediction
        const isAI = prediction === "ai";

        predictionText.textContent = isAI
            ? "AI-generated"
            : "Human-written";

        // Display confidence score
        if (
            data.confidence !== undefined &&
            data.confidence !== null
        ) {
            let confidence = Number(data.confidence);

            if (!Number.isFinite(confidence)) {
                confidenceText.textContent = "Unavailable";
            } else {
                // Convert probability from 0–1 to percentage
                if (confidence >= 0 && confidence <= 1) {
                    confidence *= 100;
                }

                confidence = Math.max(
                    0,
                    Math.min(100, confidence)
                );

                confidenceText.textContent =
                    confidence.toFixed(2) + "%";
            }
        } else {
            confidenceText.textContent = "Unavailable";
        }

        // Display text statistics
        const words = text.split(/\s+/).length;
        const characters = textInput.value.length;

        resultStats.textContent =
            `Words: ${words} | Characters: ${characters}`;

        // Apply result styling
        result.className = isAI
            ? "result ai"
            : "result human";

        result.hidden = false;

        status.textContent = "Analysis completed successfully.";

    } catch (error) {
        console.error("Prediction error:", error);

        status.textContent =
            "Unable to analyze text. " + error.message;

    } finally {
        analyzeBtn.disabled = false;
        analyzeBtn.textContent = "Detect Text";
    }
}


// Connect buttons and text input
textInput.addEventListener("input", updateCounts);

analyzeBtn.addEventListener("click", analyzeText);

clearBtn.addEventListener("click", clearText);


// Initialize the page
updateCounts();
