document.addEventListener("DOMContentLoaded", () => {
    const fileInput = document.getElementById("imageUpload");
    const gtInput = document.getElementById("groundTruthUpload");
    const fileNameDisplay = document.getElementById("fileName");
    const gtFileNameDisplay = document.getElementById("gtFileName");
    
    const originalImage = document.getElementById("originalImage");
    const reconstructedImage = document.getElementById("reconstructedImage");
    const cloudMaskImage = document.getElementById("cloudMaskImage");
    const heatmapImage = document.getElementById("heatmapImage");
    
    const statusText = document.getElementById("statusText");
    const evalType = document.getElementById("evalType");
    const cloudCoverVal = document.getElementById("cloudCoverVal");
    const confidenceVal = document.getElementById("confidenceVal");
    const maeVal = document.getElementById("maeVal");
    const psnrVal = document.getElementById("psnrVal");
    const ssimVal = document.getElementById("ssimVal");
    const statusIndicator = document.getElementById("statusIndicator");

    let selectedCloudyFile = null;
    let selectedGTFile = null;

    // Extracted prediction logic
    const runAnalysis = async () => {
        if (!selectedCloudyFile) return;

        // Reset UI processing state
        reconstructedImage.src = "https://placehold.co/600x400?text=Reconstructing...";
        cloudMaskImage.src = "https://placehold.co/600x400?text=Detecting+Mask...";
        heatmapImage.src = "https://placehold.co/600x400?text=Generating+Heatmap...";
        
        statusText.textContent = "RECONSTRUCTING";
        statusIndicator.style.backgroundColor = "#e08a3c";
        cloudCoverVal.innerHTML = "&mdash;";
        confidenceVal.innerHTML = "&mdash;";
        evalType.innerHTML = "&mdash;";
        maeVal.innerHTML = "&mdash;";
        psnrVal.innerHTML = "&mdash;";
        ssimVal.innerHTML = "&mdash;";

        const formData = new FormData();
        formData.append("image", selectedCloudyFile);
        if (selectedGTFile) {
            formData.append("ground_truth", selectedGTFile);
        }

        try {
            const response = await fetch("/predict", {
                method: "POST",
                body: formData
            });

            const data = await response.json();

            if (response.ok) {
                const cacheBuster = "?t=" + new Date().getTime();
                reconstructedImage.src = data.reconstructed_image_url + cacheBuster;
                cloudMaskImage.src = data.cloud_mask_url + cacheBuster;
                heatmapImage.src = data.heatmap_url + cacheBuster;
                
                statusText.textContent = "COMPLETE";
                statusIndicator.style.backgroundColor = "#5c9174";
                cloudCoverVal.textContent = data.cloud_cover;
                confidenceVal.textContent = data.confidence_score;
                evalType.textContent = data.evaluation_type;

                // Live Metrics
                maeVal.textContent = data.metrics.MAE;
                psnrVal.textContent = data.metrics.PSNR;
                ssimVal.textContent = data.metrics.SSIM;
            } else {
                console.error("Server Error:", data.error);
                statusText.textContent = "ERROR";
                statusIndicator.style.backgroundColor = "red";
            }

        } catch (error) {
            console.error("Network Error:", error);
            statusText.textContent = "NETWORK ERROR";
            statusIndicator.style.backgroundColor = "red";
        }
    };

    // Listeners
    gtInput.addEventListener("change", (event) => {
        const file = event.target.files[0];
        if (file) {
            selectedGTFile = file;
            gtFileNameDisplay.textContent = "Ground Truth: " + file.name;
            // If a cloudy image is already loaded, re-run the analysis to get metrics
            if (selectedCloudyFile) {
                runAnalysis();
            }
        }
    });

    fileInput.addEventListener("change", (event) => {
        const file = event.target.files[0];
        if (!file) return;

        selectedCloudyFile = file;
        fileNameDisplay.textContent = "Cloudy Image: " + file.name;
        originalImage.src = URL.createObjectURL(file);
        
        runAnalysis();
    });
});