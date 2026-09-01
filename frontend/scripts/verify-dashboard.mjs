import { fileURLToPath } from 'node:url'
import path from 'node:path'
import puppeteer from 'puppeteer-core'

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url))
const CHROME_PATH = String.raw`C:\Program Files\Google\Chrome\Application\chrome.exe`
const BASE_URL = 'http://localhost:5173'
const API_URL = 'http://localhost:8000'

// ===================== SEED REAL TIMELINE EVENTS =====================
// Customer 360's timeline is real-audit-trail-only (no fabricated events),
// so most customers show only the two inferred snapshot entries until
// something is actually run for them. Drive a few genuine backend actions
// for ONE customer here (direct API calls, not UI clicks - faster and more
// reliable) so the screenshot below shows a timeline with real, multiple
// recorded events instead of just the empty-state.
const loginResponse = await fetch(`${API_URL}/auth/login`, {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ email: 'demo-telco@churn-engine.local', password: 'DemoPass123' }),
})
const { access_token: seedToken } = await loginResponse.json()
const seedHeaders = { 'Content-Type': 'application/json', Authorization: `Bearer ${seedToken}` }

const priorityResponse = await fetch(`${API_URL}/api/priority?limit=2000`, { headers: seedHeaders })
const priorityRows = await priorityResponse.json()
const TIMELINE_TEST_CUSTOMER_ID = priorityRows[0].customerID
console.log('\n[SEED] Seeding real timeline events for customer:', TIMELINE_TEST_CUSTOMER_ID)

// /api/recommend, /api/whatif, and scenario re-scoring can only ever touch
// TEST-SET customers (same restriction as /api/priority) - so any customer
// OUTSIDE that set is guaranteed to still show the empty-state timeline,
// which is what we want to verify below alongside the populated one.
const testSetCustomerIds = new Set(priorityRows.map((row) => row.customerID))
const customersResponse = await fetch(`${API_URL}/customers`, { headers: seedHeaders })
const allCustomers = await customersResponse.json()
const EMPTY_STATE_CUSTOMER_ID = allCustomers.find((row) => !testSetCustomerIds.has(row.customer_id))?.customer_id
console.log('[SEED] Empty-state comparison customer (never touched, outside the test set):', EMPTY_STATE_CUSTOMER_ID)

await fetch(`${API_URL}/api/whatif`, {
  method: 'POST',
  headers: seedHeaders,
  body: JSON.stringify({ customer_id: TIMELINE_TEST_CUSTOMER_ID, overrides: { Contract: 'Two year' } }),
})
await fetch(`${API_URL}/api/whatif`, {
  method: 'POST',
  headers: seedHeaders,
  body: JSON.stringify({ customer_id: TIMELINE_TEST_CUSTOMER_ID, overrides: { Contract: 'Month-to-month' } }),
})
await fetch(`${API_URL}/api/whatif`, {
  method: 'POST',
  headers: seedHeaders,
  body: JSON.stringify({ customer_id: TIMELINE_TEST_CUSTOMER_ID, overrides: { MonthlyCharges: 95.5 } }),
})
// log_event=true - a plain GET without it (as PriorityTable's per-row badge
// fetches do) intentionally does NOT log, to avoid flooding every visible
// customer's timeline just from loading the Dashboard.
await fetch(`${API_URL}/api/recommend/${TIMELINE_TEST_CUSTOMER_ID}?log_event=true`, { headers: seedHeaders })
console.log('[SEED] Ran 3 what-if simulations + 1 recommendation lookup.')

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

  // Log in as the demo Telco ADMIN account, so the Admin nav link is visible.
  await page.goto(BASE_URL, { waitUntil: 'networkidle0' })
  const loginHeading = await page.$eval('h1', (el) => el.textContent)
  console.log('LOGIN PAGE HEADING:', loginHeading)

  await page.type('input[type="email"]', 'demo-telco@churn-engine.local')
  await page.type('input[type="password"]', 'DemoPass123')
  await Promise.all([
    page.click('button[type="submit"]'),
    page.waitForSelector('.dashboard-header', { timeout: 10000 }),
  ])

  // ===================== MAIN DASHBOARD =====================
  // Wait for BOTH tables (Action Queue and Priority Table) to have actually
  // rendered, not just Priority Table's badges - Action Queue's backend call
  // computes recommendations sequentially server-side and can resolve later.
  await page.waitForSelector('.recommend-badge', { timeout: 45000 }).catch(() => {})
  await page.waitForFunction(() => document.querySelectorAll('table').length >= 2, { timeout: 30000 }).catch(() => {})
  await page.waitForFunction(() => document.querySelectorAll('table tbody tr').length >= 2, { timeout: 15000 }).catch(() => {})
  await new Promise((resolve) => setTimeout(resolve, 1000))

  const navLinks = await page.$$eval('.dashboard-nav a', (els) => els.map((el) => el.textContent))
  console.log('\nNAV LINKS (should include Admin, since this is the admin demo account):', navLinks)

  const sectionTitles = await page.$$eval('.dashboard-section h2', (els) => els.map((el) => el.textContent))
  console.log('\n[MAIN] SECTIONS RENDERED:', sectionTitles)

  const businessImpactKpis = await page.$eval('.kpi-strip', (el) => el.textContent.trim()).catch(() => null)
  console.log('\n[MAIN] BUSINESS IMPACT KPI STRIP:', businessImpactKpis)

  const businessImpactCaveat = await page
    .$$eval('.dashboard-section .caveat', (els) => els.map((el) => el.textContent.trim()))
    .catch(() => [])
  console.log('[MAIN] CAVEATS FOUND:', businessImpactCaveat)

  const tableHandles = await page.$$('table')
  console.log('\n[MAIN] Number of tables found on page:', tableHandles.length)
  for (let i = 0; i < tableHandles.length; i++) {
    const headers = await tableHandles[i].$$eval('thead th', (els) => els.map((el) => el.textContent.trim()))
    const firstRow = await tableHandles[i]
      .$eval('tbody tr', (row) => row.textContent.trim())
      .catch(() => '(no rows)')
    console.log(`[MAIN] Table #${i + 1} headers:`, headers)
    console.log(`[MAIN] Table #${i + 1} first row:`, firstRow)
  }

  const healthBadges = await page
    .$$eval('.health-badge:not(.health-badge-loading)', (els) => els.slice(0, 5).map((el) => el.textContent.trim()))
    .catch(() => [])
  console.log('\n[MAIN] HEALTH SCORE BADGES (first 5):', healthBadges)

  const recommendBadges = await page
    .$$eval('.recommend-badge', (els) => els.slice(0, 5).map((el) => el.textContent.trim()))
    .catch(() => [])
  console.log('[MAIN] RECOMMENDATION BADGES (first 5):', recommendBadges)

  // Hover the Opportunity Score header to confirm the disambiguation
  // tooltip exists and is legible (a native title= attribute wouldn't be
  // screenshot-capturable, hence the custom CSS hover bubble).
  const tooltipTrigger = await page.$('.th-tooltip-trigger')
  if (tooltipTrigger) {
    await tooltipTrigger.hover()
    await new Promise((resolve) => setTimeout(resolve, 200))
    const tooltipText = await page.$eval('.th-tooltip-bubble', (el) => el.textContent.trim()).catch(() => null)
    const tooltipVisible = await page
      .$eval('.th-tooltip-bubble', (el) => window.getComputedStyle(el).display !== 'none')
      .catch(() => false)
    console.log('\n[MAIN] OPPORTUNITY SCORE TOOLTIP text:', tooltipText)
    console.log('[MAIN] OPPORTUNITY SCORE TOOLTIP visible on hover:', tooltipVisible)
    await page.screenshot({ path: path.join(SCRIPT_DIR, 'verify-screenshot-tooltip.png') })
    console.log('[MAIN] Tooltip close-up screenshot saved.')
  } else {
    console.log('\n[MAIN] WARNING: .th-tooltip-trigger not found on page.')
  }

  // Exercise the What-If simulator briefly.
  const contractBefore = await page.$eval('#whatif-contract', (el) => el.value).catch(() => null)
  if (contractBefore) {
    const target = contractBefore === 'Two year' ? 'Month-to-month' : 'Two year'
    await page.select('#whatif-contract', target)
    await page.click('.whatif-controls button')
    await page.waitForSelector('.whatif-result', { timeout: 10000 }).catch(() => {})
    await new Promise((resolve) => setTimeout(resolve, 500))
    const whatifResultText = await page.$eval('.whatif-result', (el) => el.textContent.trim()).catch(() => null)
    console.log('\n[MAIN] WHAT-IF RESULT:', whatifResultText)
  }

  const mainNotAvailable = await page.$$eval('.not-available', (els) => els.map((el) => el.textContent))
  const mainErrors = await page.$$eval('.error-text', (els) => els.map((el) => el.textContent))
  console.log('\n[MAIN] NOT-AVAILABLE BLOCKS:', mainNotAvailable)
  console.log('[MAIN] ERROR BLOCKS:', mainErrors)

  await page.mouse.move(0, 0)
  await page.screenshot({ path: path.join(SCRIPT_DIR, 'verify-screenshot-main.png'), fullPage: true })
  console.log('\n[MAIN] Screenshot saved.')

  // ===================== ADMIN VIEW =====================
  await page.click('.dashboard-nav a[href="/admin"]')
  await page.waitForSelector('.dashboard-header', { timeout: 10000 })
  await new Promise((resolve) => setTimeout(resolve, 3000))

  const adminUrl = page.url()
  console.log('\n[ADMIN] URL:', adminUrl)

  const adminSectionTitles = await page.$$eval('.dashboard-section h2', (els) => els.map((el) => el.textContent))
  console.log('[ADMIN] SECTIONS RENDERED:', adminSectionTitles)

  const adminKpiText = await page.$eval('.kpi-strip', (el) => el.textContent.trim()).catch(() => null)
  console.log('[ADMIN] KPI STRIP (should show ROC-AUC/PR-AUC):', adminKpiText)

  await page.waitForSelector('.alerts-summary-chip', { timeout: 15000 }).catch(() => {})
  const adminAlertsSummary = await page
    .$$eval('.alerts-summary-chip', (els) => els.map((el) => el.textContent.trim()))
    .catch(() => [])
  console.log('[ADMIN] ALERTS SUMMARY CHIPS:', adminAlertsSummary)

  const adminSegmentLabels = await page
    .$$eval('.card-grid .info-card h3', (els) => els.map((el) => el.textContent))
    .catch(() => [])
  console.log('[ADMIN] SEGMENT LABELS:', adminSegmentLabels)

  await page.screenshot({ path: path.join(SCRIPT_DIR, 'verify-screenshot-admin.png'), fullPage: true })
  console.log('[ADMIN] Screenshot saved.')

  // ===================== EXECUTIVE VIEW =====================
  await page.click('.dashboard-nav a[href="/executive"]')
  await page.waitForSelector('.dashboard-header', { timeout: 10000 })
  // /api/action-queue computes recommendations sequentially server-side
  // (~0.15-0.5s per customer x limit) - wait for actual content, not a guess.
  await page.waitForSelector('.opportunity-list li', { timeout: 30000 }).catch(() => {})
  await new Promise((resolve) => setTimeout(resolve, 500))

  const executiveUrl = page.url()
  console.log('\n[EXECUTIVE] URL:', executiveUrl)

  const executiveSectionTitles = await page.$$eval('.dashboard-section h2', (els) => els.map((el) => el.textContent))
  console.log('[EXECUTIVE] SECTIONS RENDERED:', executiveSectionTitles)

  const executiveKpiText = await page.$eval('.kpi-strip', (el) => el.textContent.trim()).catch(() => null)
  console.log('[EXECUTIVE] KPI STRIP:', executiveKpiText)

  const opportunityListItems = await page
    .$$eval('.opportunity-list li', (els) => els.map((el) => el.textContent.trim()))
    .catch(() => [])
  console.log('[EXECUTIVE] TOP OPPORTUNITIES (count=' + opportunityListItems.length + '):', opportunityListItems.slice(0, 3))

  await page.screenshot({ path: path.join(SCRIPT_DIR, 'verify-screenshot-executive.png'), fullPage: true })
  console.log('[EXECUTIVE] Screenshot saved.')

  // ===================== SCENARIO SIMULATOR =====================
  await page.click('.dashboard-nav a[href="/scenarios"]')
  await page.waitForSelector('.scenario-builder', { timeout: 10000 })

  const scenarioUrl = page.url()
  console.log('\n[SCENARIO] URL:', scenarioUrl)

  // --- Run scenario #1: price increase (uniform_charge_change) ---
  await page.select('#scenario-type', 'uniform_charge_change')
  await page.type('#scenario-name', 'Verify: 10% price increase')
  // The percent-change range input defaults to 10, which is what we want -
  // just click Run.
  await page.click('.scenario-builder button[type="submit"]')
  console.log('[SCENARIO] Price-increase scenario submitted, waiting for results...')
  await page.waitForFunction(
    () => document.querySelector('.scenario-summary') && document.querySelector('.scenario-summary').textContent.length > 0,
    { timeout: 30000 },
  )
  const priceScenarioSummary = await page.$eval('.scenario-summary', (el) => el.textContent.trim())
  console.log('[SCENARIO] Price-increase summary:', priceScenarioSummary)
  const priceScenarioKpis = await page.$$eval('.scenario-results .kpi-value', (els) => els.map((el) => el.textContent.trim()))
  console.log('[SCENARIO] Price-increase KPI values:', priceScenarioKpis)

  await page.screenshot({ path: path.join(SCRIPT_DIR, 'verify-screenshot-scenario-price.png'), fullPage: true })
  console.log('[SCENARIO] Price-increase screenshot saved.')

  // --- Run scenario #2: favorable contract migration ---
  await page.select('#scenario-type', 'contract_migration')
  await page.evaluate(() => {
    document.getElementById('scenario-name').value = ''
  })
  await page.type('#scenario-name', 'Verify: M2M to Two-year migration')
  await page.select('#scenario-from-contract', 'Month-to-month')
  await page.select('#scenario-to-contract', 'Two year')
  await page.click('.scenario-builder button[type="submit"]')
  console.log('[SCENARIO] Contract-migration scenario submitted, waiting for results...')
  await page.waitForFunction(
    () => {
      const el = document.querySelector('.scenario-summary')
      return el && el.textContent.includes('Migrating')
    },
    { timeout: 30000 },
  )
  const migrationScenarioSummary = await page.$eval('.scenario-summary', (el) => el.textContent.trim())
  console.log('[SCENARIO] Contract-migration summary:', migrationScenarioSummary)

  await new Promise((resolve) => setTimeout(resolve, 500))
  await page.waitForFunction(() => document.querySelectorAll('.scenario-history-item').length >= 2, { timeout: 15000 })
  const historyItems = await page.$$eval('.scenario-history-item', (els) => els.map((el) => el.textContent.trim()))
  console.log('[SCENARIO] History list entries (count=' + historyItems.length + '):', historyItems)

  await page.screenshot({ path: path.join(SCRIPT_DIR, 'verify-screenshot-scenario-migration.png'), fullPage: true })
  console.log('[SCENARIO] Contract-migration + history screenshot saved.')

  // --- Compare the two saved scenarios side by side ---
  const historyCheckboxes = await page.$$('.scenario-history-checkbox')
  await historyCheckboxes[0].click()
  await historyCheckboxes[1].click()
  const compareButtonHandle = await page.$$('.scenario-history > button')
  await compareButtonHandle[compareButtonHandle.length - 1].click()
  await page.waitForSelector('.scenario-compare-columns', { timeout: 10000 })
  await new Promise((resolve) => setTimeout(resolve, 500))

  const compareColumnTitles = await page.$$eval('.scenario-compare-column h3', (els) => els.map((el) => el.textContent.trim()))
  console.log('[SCENARIO] Compare view column titles:', compareColumnTitles)

  await page.screenshot({ path: path.join(SCRIPT_DIR, 'verify-screenshot-scenario-compare.png'), fullPage: true })
  console.log('[SCENARIO] Compare screenshot saved.')

  // ===================== CUSTOMER 360 =====================
  await page.click('.dashboard-nav a[href="/customer-360"]')
  await page.waitForSelector('#customer-360-select', { timeout: 10000 })

  await page.select('#customer-360-select', TIMELINE_TEST_CUSTOMER_ID)
  await page.waitForFunction(() => document.querySelectorAll('.timeline-event').length >= 3, { timeout: 15000 })
  await new Promise((resolve) => setTimeout(resolve, 500))

  const timelineEventTypes = await page.$$eval('.timeline-type', (els) => els.map((el) => el.textContent.trim()))
  console.log('\n[CUSTOMER 360] Timeline event types (chronological):', timelineEventTypes)

  const timelineBadges = await page.$$eval('.timeline-badge', (els) => els.map((el) => el.textContent.trim()))
  console.log('[CUSTOMER 360] Timeline badges:', timelineBadges)

  const recordedCount = timelineBadges.filter((b) => b === 'recorded').length
  console.log('[CUSTOMER 360] Real ("recorded") events found:', recordedCount)

  await page.screenshot({ path: path.join(SCRIPT_DIR, 'verify-screenshot-customer360-timeline.png'), fullPage: true })
  console.log('[CUSTOMER 360] Timeline screenshot saved.')

  // Also verify the empty-state message on a customer with no real events.
  if (EMPTY_STATE_CUSTOMER_ID) {
    await page.select('#customer-360-select', EMPTY_STATE_CUSTOMER_ID)
    await page.waitForFunction(() => document.querySelectorAll('.timeline-event').length >= 1, { timeout: 15000 })
    await new Promise((resolve) => setTimeout(resolve, 500))
    const emptyStateEventTypes = await page.$$eval('.timeline-type', (els) => els.map((el) => el.textContent.trim()))
    const emptyStateText = await page.$eval('.caveat', (el) => el.textContent.trim()).catch(() => null)
    console.log('[CUSTOMER 360] Empty-state customer timeline event types:', emptyStateEventTypes)
    console.log('[CUSTOMER 360] Empty-state message for a thin-history customer:', emptyStateText)
    await page.screenshot({ path: path.join(SCRIPT_DIR, 'verify-screenshot-customer360-empty-state.png'), fullPage: true })
    console.log('[CUSTOMER 360] Empty-state screenshot saved.')
  }

  console.log('\nBROWSER CONSOLE ERRORS (across whole session):', consoleErrors.length ? consoleErrors : 'none')
} finally {
  await browser.close()
}
