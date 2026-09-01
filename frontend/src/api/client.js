import axios from 'axios'

// Build-time override (see frontend/Dockerfile's VITE_API_BASE_URL build
// arg / docker-compose.yml) for deployments where the frontend and backend
// aren't both reachable at localhost - e.g. real domains, or a Docker host
// where the backend's mapped port differs from 8000. Falls back to the
// existing local-dev default when unset, so `npm run dev` behavior is
// unchanged.
const client = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000',
})

let onUnauthorized = () => {}

export function setUnauthorizedHandler(handler) {
  onUnauthorized = handler
}

export function setAuthToken(token) {
  if (token) {
    client.defaults.headers.common.Authorization = `Bearer ${token}`
  } else {
    delete client.defaults.headers.common.Authorization
  }
}

client.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response && error.response.status === 401) {
      onUnauthorized()
    }
    return Promise.reject(error)
  },
)

export default client
