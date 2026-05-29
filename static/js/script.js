const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const captureBtn = document.getElementById('capture-btn');
const clearBtn = document.getElementById('clear-btn');
const result = document.getElementById('result');
const resultBadge = document.getElementById('result-badge');
const loadingSpinner = document.getElementById('loading');
const cameraStatus = document.getElementById('camera-status');
const confidenceText = document.getElementById('confidence-text');
const confidenceFill = document.getElementById('confidence-fill');
const statDigit = document.getElementById('stat-digit');
const statConfidence = document.getElementById('stat-confidence');
const previewCard = document.getElementById('preview-card');
const previewImage = document.getElementById('capture-preview');

const ctx = canvas.getContext('2d');

// Start camera with better error handling
async function startCamera() {
  try {
    // Check if getUserMedia is supported
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error('Your browser does not support camera access');
    }

    // Request camera access with proper constraints
    const constraints = {
      video: {
        width: { ideal: 280 },
        height: { ideal: 280 },
        facingMode: 'user'
      },
      audio: false
    };

    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    video.srcObject = stream;
    
    // Wait for video to start playing before enabling capture
    video.onloadedmetadata = () => {
      video.play();
      captureBtn.disabled = false;
      clearBtn.disabled = false;
      cameraStatus.textContent = '✓ Camera is ready!';
      cameraStatus.className = 'camera-status success';
    };

  } catch (err) {
    console.error('Camera error:', err);
    captureBtn.disabled = true;
    clearBtn.disabled = true;
    
    let errorMessage = '';
    
    if (err.name === 'NotAllowedError') {
      errorMessage = '❌ Camera access was denied. Please allow camera permissions.';
    } else if (err.name === 'NotFoundError') {
      errorMessage = '❌ No camera device found on this computer.';
    } else if (err.name === 'NotReadableError') {
      errorMessage = '❌ Camera is being used by another application.';
    } else if (err.name === 'OverconstrainedError') {
      errorMessage = '❌ Your camera does not meet the requirements.';
    } else {
      errorMessage = '❌ ' + (err.message || 'Failed to access camera');
    }
    
    cameraStatus.textContent = errorMessage;
    cameraStatus.className = 'camera-status error';
    result.textContent = '?';
  }
}

// Initialize camera on page load
window.addEventListener('DOMContentLoaded', startCamera);

// Capture and predict
captureBtn.addEventListener('click', () => {
  if (!video.srcObject) {
    cameraStatus.textContent = '❌ Camera is not ready. Please refresh the page.';
    cameraStatus.className = 'camera-status error';
    return;
  }

  loadingSpinner.classList.remove('d-none');
  resultBadge.classList.add('d-none');

  // Set canvas to a square and capture only the centered frame area
  const targetSize = 280;
  canvas.width = targetSize;
  canvas.height = targetSize;

  // Calculate centered square crop from the video feed
  const vidW = video.videoWidth || targetSize;
  const vidH = video.videoHeight || targetSize;
  const size = Math.min(vidW, vidH);
  const sx = Math.max(0, (vidW - size) / 2);
  const sy = Math.max(0, (vidH - size) / 2);

  // drawImage(source, sx, sy, sWidth, sHeight, dx, dy, dWidth, dHeight)
  ctx.drawImage(video, sx, sy, size, size, 0, 0, targetSize, targetSize);
  const dataURL = canvas.toDataURL('image/png');

  fetch('/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image: dataURL })
  })
    .then(res => res.json())
    .then(data => {
      loadingSpinner.classList.add('d-none');
      
      if (data.error) {
        throw new Error(data.error);
      }
      
      const digit = data.digit;
      const confidence = (data.confidence * 100).toFixed(1);
      
      // Update prediction badge
      result.textContent = digit;
      confidenceText.textContent = confidence + '%';
      confidenceFill.style.width = confidence + '%';
      
      // Update stats
      statDigit.textContent = digit;
      statConfidence.textContent = confidence + '%';
      
      // Show badge
      resultBadge.classList.remove('d-none');
      cameraStatus.textContent = '✓ Prediction successful!';
      cameraStatus.className = 'camera-status success';
    })
    .catch(err => {
      console.error('Prediction error:', err);
      loadingSpinner.classList.add('d-none');
      result.textContent = '?';
      resultBadge.classList.remove('d-none');
      cameraStatus.textContent = '❌ Prediction failed: ' + err.message;
      cameraStatus.className = 'camera-status error';
    });
});

// Clear results
clearBtn.addEventListener('click', () => {
  result.textContent = '?';
  resultBadge.classList.add('d-none');
  confidenceText.textContent = '0%';
  confidenceFill.style.width = '0%';
  statDigit.textContent = '—';
  statConfidence.textContent = '0%';
  previewImage.src = '';
  previewCard.classList.add('d-none');
  cameraStatus.textContent = 'Point your digit inside the glowing frame and tap capture.';
  cameraStatus.className = 'camera-status';
});
