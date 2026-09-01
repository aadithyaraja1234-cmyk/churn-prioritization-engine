import { useEffect, useState } from 'react'
import client from '../api/client'

export function useApiData(url, params) {
  const [data, setData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const paramsKey = params ? JSON.stringify(params) : ''

  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setError(null)
    client
      .get(url, { params })
      .then((response) => {
        if (!cancelled) setData(response.data)
      })
      .catch((err) => {
        if (!cancelled) setError(err.response?.data?.detail || err.message)
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [url, paramsKey])

  return { data, loading, error }
}
