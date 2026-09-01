// Real, live verification of the async background training-job
// infrastructure, isolated from the rest of company onboarding: registers a
// fresh tenant, walks the onboarding wizard to completion, clicks
// "Start Training", and watches the real GET /api/training/status/{job_id}
// polling indicator move from queued to running to succeeded -
// screenshotting each state as it's observed. Training itself is real
// (src.models.train.train_model(), the same trainer Telco/Banking use) -
// this only checks the job lifecycle (queued/running/succeeded transitions,
// polling, status indicator text), not the resulting metrics themselves;
// see frontend/scripts/verify-onboarding-demo-companies.mjs for full
// onboarding coverage across multiple companies/data shapes.
import { fileURLToPath } from 'node:url'
import path from 'node:path'
import fs from 'node:fs'
import puppeteer from 'puppeteer-core'

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url))
const CHROME_PATH = String.raw`C:\Program Files\Google\Chrome\Application\chrome.exe`
const BASE_URL = 'http://localhost:5173'
const REPO_ROOT = path.resolve(SCRIPT_DIR, '..', '..')

const RUN_ID = Date.now().toString(36)
const COMPANY_NAME = `Training Verify Co ${RUN_ID}`
const EMAIL = `training-verify-${RUN_ID}@example.com`
const PASSWORD = 'Correct-Horse-9!'

function shot(page, name) {
  return page.screenshot({ path: path.join(SCRIPT_DIR, `verify-training-${name}.png`) })
}

async function clearAndType(page, selector, text) {
  await page.click(selector)
  await page.keyboard.down('Control')
  await page.keyboard.press('KeyA')
  await page.keyboard.up('Control')
  await page.keyboard.press('Backspace')
  await page.type(selector, text)
}

const browser = await puppeteer.launch({
  executablePath: CHROME_PATH,
  headless: true,
  args: ['--no-sandbox'],
})

try {
  const page = await browser.newPage()
  await page.setViewport({ width: 1200, height: 900 })
  const consoleErrors = []
  page.on('console', (msg) => {
    if (msg.type() === 'error') consoleErrors.push(msg.text())
  })
  page.on('pageerror', (err) => consoleErrors.push(String(err)))

  // ===================== REGISTER =====================
  await page.goto(`${BASE_URL}/register`, { waitUntil: 'networkidle0' })
  await clearAndType(page, 'input[autocomplete="organization"]', COMPANY_NAME)
  await clearAndType(page, 'input[autocomplete="username"]', EMAIL)
  await clearAndType(page, 'input[autocomplete="new-password"]', PASSWORD)
  await Promise.all([
    page.click('button[type="submit"]'),
    page.waitForFunction(() => location.pathname === '/welcome', { timeout: 10000 }),
  ])
  console.log('[REGISTER] Registered', COMPANY_NAME)

  // ===================== UPLOAD + MAP + VALIDATE + CONFIRM =====================
  await page.click('.register-welcome-button-primary')
  await page.waitForFunction(() => location.pathname === '/data-onboarding', { timeout: 10000 })
  await page.waitForSelector('.onboarding-progress', { timeout: 10000 })

  const csvPath = path.join(REPO_ROOT, 'data', 'test_onboarding_samples', 'company_meridian_wireless.csv')
  if (!fs.existsSync(csvPath)) throw new Error(`Fixture CSV not found: ${csvPath}`)
  const fileInput = await page.$('input[type="file"]')
  await fileInput.uploadFile(csvPath)

  await page.waitForSelector('.mapping-table', { timeout: 15000 })
  console.log('[WIZARD] Uploaded and mapped (Meridian Wireless - clean data, auto-suggests correctly).')

  await Promise.all([
    page.click('.onboarding-actions button:not([disabled]):last-child'),
    page.waitForSelector('.verdict-banner', { timeout: 15000 }),
  ])
  const verdictText = await page.$eval('.verdict-banner', (el) => el.textContent.trim())
  console.log('[WIZARD] Validation verdict:', verdictText)

  await page.click('.onboarding-actions button:not(:disabled):last-child') // "Continue"
  await page.waitForFunction(
    () => document.body.textContent.includes('Your workspace is marked as ready'),
    { timeout: 10000 },
  )
  console.log('[WIZARD] Onboarding confirmed - ready to start training.')
  await shot(page, '1-ready-to-start')

  // ===================== START TRAINING =====================
  await page.waitForSelector('.onboarding-actions button', { timeout: 5000 })
  const startButtonText = await page.$eval(
    '.onboarding-actions button',
    (el) => el.textContent.trim(),
  )
  console.log('\n[TRAINING] Button found:', startButtonText)

  const startedAt = Date.now()
  await page.click('.onboarding-actions button') // "Start Training"
  await page.waitForSelector('.training-status', { timeout: 10000 })
  const elapsedToButtonReturn = Date.now() - startedAt
  console.log(`[TRAINING] "Start Training" click -> indicator appeared in ${elapsedToButtonReturn}ms (real training runs server-side - a few seconds for this fixture's 6,000 rows).`)

  // ===================== OBSERVE THE REAL STATUS TRANSITIONS =====================
  const seenStatuses = []
  let lastStatus = null
  const deadline = Date.now() + 60000
  let shotIndex = 2
  while (Date.now() < deadline) {
    const statusText = await page
      .$eval('.training-status-row span:last-child', (el) => el.textContent.trim())
      .catch(() => null)
    const statusClass = await page.$eval('.training-status', (el) => el.className).catch(() => null)
    if (statusText && statusText !== lastStatus) {
      lastStatus = statusText
      seenStatuses.push({ t: Date.now() - startedAt, statusText, statusClass })
      console.log(`[TRAINING] t=+${Date.now() - startedAt}ms status indicator now reads: "${statusText}" (${statusClass})`)
      await shot(page, `${shotIndex}-${statusText.replace(/[^a-z0-9]+/gi, '-').toLowerCase()}`)
      shotIndex += 1
    }
    if (statusClass && (statusClass.includes('training-status-succeeded') || statusClass.includes('training-status-failed'))) {
      break
    }
    await new Promise((resolve) => setTimeout(resolve, 300))
  }

  const finalNote = await page.$eval('.training-status-note', (el) => el.textContent.trim()).catch(() => null)
  console.log('\n[TRAINING] Final note shown in UI:', finalNote)
  await shot(page, `${shotIndex}-final`)

  console.log('\n[SUMMARY] All observed status texts, in order:', seenStatuses.map((s) => s.statusText))
  console.log('[CONSOLE ERRORS]:', consoleErrors.length ? consoleErrors : 'none')
  console.log('\nDONE. Screenshots written to', SCRIPT_DIR)
} finally {
  await browser.close()
}
