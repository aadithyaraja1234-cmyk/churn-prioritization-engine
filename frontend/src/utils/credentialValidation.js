// Mirrors src/validation/credentials.py's policy for real-time client-side
// feedback. The backend is the actual source of truth (this file has no
// authority - it exists only so a user sees feedback before submitting) -
// if that policy changes, this file must change with it.

export const MIN_PASSWORD_LENGTH = 10
export const MIN_PASSWORD_CHARACTER_CLASSES = 3
export const MIN_COMPANY_NAME_LENGTH = 2
export const MAX_COMPANY_NAME_LENGTH = 100

const EMAIL_PATTERN = /^[^@\s]+@[^@\s]+\.[^@\s]+$/

export function passwordCharacterClassCount(password) {
  return [/[a-z]/, /[A-Z]/, /[0-9]/, /[^a-zA-Z0-9]/].filter((pattern) => pattern.test(password)).length
}

// 0-4 scale for the strength meter - not the pass/fail rule itself (see
// passwordError below for that), just a smoother visual signal.
export function passwordStrengthScore(password) {
  if (!password) return 0
  let score = 0
  if (password.length >= MIN_PASSWORD_LENGTH) score += 1
  if (password.length >= 14) score += 1
  score += Math.max(0, passwordCharacterClassCount(password) - 1)
  return Math.min(score, 4)
}

export function passwordError(password) {
  if (!password || password.length < MIN_PASSWORD_LENGTH) {
    return `Password must be at least ${MIN_PASSWORD_LENGTH} characters.`
  }
  if (passwordCharacterClassCount(password) < MIN_PASSWORD_CHARACTER_CLASSES) {
    return `Password must include at least ${MIN_PASSWORD_CHARACTER_CLASSES} of: lowercase letters, uppercase letters, numbers, symbols.`
  }
  return null
}

export function emailError(email) {
  if (!email || !EMAIL_PATTERN.test(email.trim())) {
    return 'Enter a valid email address.'
  }
  return null
}

export function companyNameError(name) {
  const trimmed = (name || '').trim()
  if (trimmed.length < MIN_COMPANY_NAME_LENGTH) {
    return `Company name must be at least ${MIN_COMPANY_NAME_LENGTH} characters.`
  }
  if (trimmed.length > MAX_COMPANY_NAME_LENGTH) {
    return `Company name must be under ${MAX_COMPANY_NAME_LENGTH} characters.`
  }
  return null
}
