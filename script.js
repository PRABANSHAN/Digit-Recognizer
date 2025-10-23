const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const captureBtn = document.getElementById('capture-btn');
const clearBtn = document.getElementById('clear-btn');
const result = document.getElementById('result');
const loadingSpinner = document.getElementById('loading');

const ctx = canvas.getContext('2d');

// Start webcam
navigator.mediaDevices.getUserMedia({ video: { width: 280, height: 280 } })
  .then(stream => {
    video.srcObject = stream;
  })
  .catch(err => {
    alert('Could not access webcam: ' + err);
  });

captureBtn.addEventListener('click', () => {
  loadingSpinner.style.display = 'block';
  result.textContent = '';

  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  const dataURL = canvas.toDataURL('image/png');

  fetch('/predict', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ image: dataURL })
  })
    .then(res => res.json())
    .then(data => {
      loadingSpinner.style.display = 'none';
      result.textContent = `Predicted Digit: ${data.digit}`;
    })
    .catch(() => {
      loadingSpinner.style.display = 'none';
      alert('Prediction failed. Please try again.');
    });
});

clearBtn.addEventListener('click', () => {
  result.textContent = '';
});
