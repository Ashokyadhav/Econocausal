const API_BASE_URL = 'http://127.0.0.1:8000/api'

async function handleResponse(response) {
  if (!response.ok) {
    let message = `Request failed: ${response.status}`

    try {
      const error = await response.json()
      message = error.detail || message
    } catch {
      // Keep default error message.
    }

    throw new Error(
      typeof message === 'string' ? message : JSON.stringify(message),
    )
  }

  return response
}

export async function uploadDataset(file) {
  const formData = new FormData()
  formData.append('file', file)

  const response = await fetch(`${API_BASE_URL}/analytics/upload`, {
    method: 'POST',
    body: formData,
  })

  await handleResponse(response)
  return response.json()
}

export async function trainModel(datasetId) {
  const response = await fetch(
    `${API_BASE_URL}/analytics/${encodeURIComponent(datasetId)}/train`,
    { method: 'POST' },
  )

  await handleResponse(response)
  return response.json()
}

export async function getResults(datasetId) {
  const response = await fetch(
    `${API_BASE_URL}/analytics/${encodeURIComponent(datasetId)}/results`,
  )

  await handleResponse(response)
  return response.json()
}

export async function exportResults(datasetId) {
  const response = await fetch(
    `${API_BASE_URL}/analytics/${encodeURIComponent(datasetId)}/export`,
  )

  await handleResponse(response)
  return response.blob()
}
