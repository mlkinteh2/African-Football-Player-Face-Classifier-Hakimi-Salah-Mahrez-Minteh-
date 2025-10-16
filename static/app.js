const input = document.getElementById('file-input')
const btn = document.getElementById('predict-btn')
const resultDiv = document.getElementById('result')
const faceImg = document.getElementById('face-img')

// If you serve this static page from Live Server or another origin,
// set the API_URL to the Flask backend. The Flask server defaults to http://127.0.0.1:5000
const API_URL = 'http://127.0.0.1:5000/predict'

btn.addEventListener('click', async () => {
  resultDiv.innerHTML = ''
  if (!input.files || input.files.length === 0) {
    resultDiv.textContent = 'Please choose an image first.'
    return
  }

  const file = input.files[0]
  const form = new FormData()
  form.append('image', file)

  resultDiv.textContent = 'Predicting...'

  try {
    const res = await fetch(API_URL, { method: 'POST', body: form })
    const text = await res.text()
    let data = null
    try {
      data = JSON.parse(text)
    } catch (e) {
      resultDiv.textContent = `Server returned non-JSON response (status ${res.status}): ${text}`
      return
    }
    if (!res.ok || data.error) {
      if (res.status === 503) {
        resultDiv.textContent = 'Model is still loading on the server. Please wait a few seconds and try again.'
      } else {
        resultDiv.textContent = data.error || `Prediction failed (status ${res.status})`
      }
      return
    }

    // show face preview
    if (data.face_image) {
      faceImg.src = 'data:image/jpeg;base64,' + data.face_image
      faceImg.style.display = 'block'
    }

    // show results
    resultDiv.innerHTML = `<h3>Prediction: ${data.player_name || 'Unknown'}</h3>`
    if (data.detection) {
      const det = document.createElement('div')
      det.textContent = `Detection: ${data.detection}`
      resultDiv.appendChild(det)
    }

    // Populate probability sidebar
    const probs = data.probabilities || {}
    const probList = document.getElementById('prob-list')
    probList.innerHTML = ''
    Object.keys(probs).sort((a,b)=>probs[b]-probs[a]).forEach(k=>{
      const row = document.createElement('div')
      row.className = 'prob-row'
      const nameDiv = document.createElement('div')
      nameDiv.className = 'prob-name'
      nameDiv.textContent = k
      const valDiv = document.createElement('div')
      valDiv.className = 'prob-value'
      valDiv.textContent = (probs[k]).toFixed(2) + '%'
      row.appendChild(nameDiv)
      row.appendChild(valDiv)
      probList.appendChild(row)
    })

    // Also, update a simple result caption under face preview
    const caption = document.createElement('div')
    caption.className = 'result-caption'
    caption.innerHTML = `<strong>${data.player_name || 'Unknown'}</strong>`
    resultDiv.appendChild(caption)

  } catch (err) {
    console.error(err)
    resultDiv.textContent = 'Network or server error: ' + err.message
  }
})
