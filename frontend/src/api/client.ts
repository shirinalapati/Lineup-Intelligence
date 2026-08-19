import type { Unavailable } from './types'
import { isUnavailable } from './types'

export class ApiError extends Error {
  status: number
  body: unknown

  constructor(message: string, status: number, body: unknown) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.body = body
  }
}

export async function apiGet<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: {
      Accept: 'application/json',
      ...(init?.headers ?? {}),
    },
  })
  const body = await res.json().catch(() => null)
  if (!res.ok) {
    const detail =
      body && typeof body === 'object' && 'detail' in body
        ? String((body as { detail: unknown }).detail)
        : res.statusText
    throw new ApiError(detail || `Request failed (${res.status})`, res.status, body)
  }
  return body as T
}

export async function apiPost<T>(
  path: string,
  payload: unknown,
  init?: RequestInit,
): Promise<T> {
  const res = await fetch(path, {
    method: 'POST',
    ...init,
    headers: {
      Accept: 'application/json',
      'Content-Type': 'application/json',
      ...(init?.headers ?? {}),
    },
    body: JSON.stringify(payload),
  })
  const body = await res.json().catch(() => null)
  if (!res.ok) {
    const detail =
      body && typeof body === 'object' && 'detail' in body
        ? String((body as { detail: unknown }).detail)
        : res.statusText
    throw new ApiError(detail || `Request failed (${res.status})`, res.status, body)
  }
  return body as T
}

export function asUnavailable(reason: string): Unavailable {
  return { available: false, reason }
}

export function reasonOf(v: unknown, fallback = 'Not available'): string {
  if (isUnavailable(v)) return v.reason || fallback
  return fallback
}
