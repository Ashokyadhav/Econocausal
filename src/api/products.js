const API_BASE_URL = 'http://127.0.0.1:8000/api'

async function request(endpoint, options = {}) {
  const response = await fetch(`${API_BASE_URL}${endpoint}`, {
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
    ...options,
  })

  if (!response.ok) {
    let message = `Request failed: ${response.status}`

    try {
      const error = await response.json()
      message = error.detail || message
    } catch {
      // Keep the default error message.
    }

    throw new Error(
      typeof message === 'string' ? message : JSON.stringify(message),
    )
  }

  if (response.status === 204) {
    return null
  }

  return response.json()
}

export function getProducts(search = '') {
  const query = search.trim()
    ? `?search=${encodeURIComponent(search.trim())}`
    : ''

  return request(`/products${query}`)
}

export function createProduct(product) {
  return request('/products', {
    method: 'POST',
    body: JSON.stringify(product),
  })
}

export function updateProduct(id, product) {
  return request(`/products/${id}`, {
    method: 'PUT',
    body: JSON.stringify(product),
  })
}

export function deleteProduct(id) {
  return request(`/products/${id}`, {
    method: 'DELETE',
  })
}