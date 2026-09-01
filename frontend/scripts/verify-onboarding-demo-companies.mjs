// Real, live walkthrough of all 5 onboarding demo companies (see
// data/test_onboarding_samples/README.md) against the actual running
// backend + frontend - registers a fresh tenant per company, uploads that
// company's real CSV through the real upload endpoint, drives the mapping
// UI, and captures the real validation report. Nothing here is simulated:
// every result printed is what the live API actually returned.
import { fileURLToPath } from 'node:url'
import path from 'node:path'
import fs from 'node:fs'
import puppeteer from 'puppeteer-core'

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url))
const CHROME_PATH = String.raw`C:\Program Files\Google\Chrome\Application\chrome.exe`
const BASE_URL = 'http://localhost:5173'
const REPO_ROOT = path.resolve(SCRIPT_DIR, '..', '..')
const SAMPLES_DIR = path.join(REPO_ROOT, 'data', 'test_onboarding_samples')
const PASSWORD = 'Onboard-Demo-2026!'

async function clearAndType(page, selector, text) {
  await page.click(selector)
  await page.keyboard.down('Control')
  await page.keyboard.press('KeyA')
  await page.keyboard.up('Control')
  await page.keyboard.press('Backspace')
  await page.type(selector, text)
}

async function setMappingRole(page, columnName, role) {
  const rowIndex = await page.evaluate((column) => {
    const rows = [...document.querySelectorAll('.mapping-table tbody tr')]
    return rows.findIndex((r) => r.children[0].textContent.trim() === column)
  }, columnName)
  if (rowIndex === -1) throw new Error(`Column not found in mapping table: ${columnName}`)
  await page.select(`.mapping-table tbody tr:nth-child(${rowIndex + 1}) select`, role)
}

const COMPANIES = [
  {
    name: 'Meridian Wireless',
    email: 'admin@meridianwireless.demo',
    csv: 'company_meridian_wireless.csv',
    expectRejected: false,
    // customerID/Churn/MonthlyCharges already auto-suggest correctly by
    // exact schema-field name match - no manual mapping fixes needed.
    mappingFixes: [],
  },
  {
    name: 'Harborline Insurance Partners',
    email: 'admin@harborline-insurance.demo',
    csv: 'company_harborline_insurance.csv',
    expectRejected: false,
    mappingFixes: [
      ['PolicyholderID', 'customer_id'],
      ['AnnualPremiumUSD', 'revenue'],
      ['Lapsed', 'target'],
    ],
  },
  {
    name: 'Fernwood Retail Collective',
    email: 'admin@fernwood-retail.demo',
    csv: 'company_fernwood_retail.csv',
    expectRejected: false,
    // cust_ref / mo_bill_usd / left_us get NO auto-suggestion at all
    // (below the 0.72 similarity threshold - default to "ignore"), and
    // agreement_type gets a WRONG low-confidence auto-suggestion (matched
    // to PaymentMethod, 0.75, when it's really Contract) - all confirmed
    // against the real suggest_mapping() output before writing this script.
    mappingFixes: [
      ['cust_ref', 'customer_id'],
      ['mo_bill_usd', 'revenue'],
      ['left_us', 'target'],
      // agreement_type's suggested ROLE ("feature") is already correct even
      // though its suggested field label is wrong - left as-is deliberately,
      // see the writeup for what that means for missing_core_fields.
    ],
  },
  {
    name: 'Cobalt Health Network',
    email: 'admin@cobalt-health.demo',
    csv: 'company_cobalt_health.csv',
    expectRejected: false,
    // Same column names as Meridian - all auto-suggest correctly by name.
    // The problem here is invisible at mapping time (nulls/constants),
    // which is the point.
    mappingFixes: [],
  },
  {
    name: 'Redline Motors Direct',
    email: 'admin@redline-motors.demo',
    csv: 'company_redline_motors.csv',
    expectRejected: true,
    mappingFixes: [],
  },
]

const results = []

const browser = await puppeteer.launch({
  executablePath: CHROME_PATH,
  headless: true,
  args: ['--no-sandbox'],
})

// Optional CLI filter, e.g. `node verify-onboarding-demo-companies.mjs cobalt`
// - runs only companies whose name matches, for re-verifying one company
// after a backend change without re-running the whole demo.
const filter = process.argv[2]?.toLowerCase()
const companiesToRun = filter ? COMPANIES.filter((c) => c.name.toLowerCase().includes(filter)) : COMPANIES

try {
  for (const company of companiesToRun) {
    console.log(`\n${'='.repeat(70)}\n${company.name}\n${'='.repeat(70)}`)
    const page = await browser.newPage()
    await page.setViewport({ width: 1200, height: 900 })
    const consoleErrors = []
    page.on('console', (msg) => {
      if (msg.type() === 'error') consoleErrors.push(msg.text())
    })
    page.on('pageerror', (err) => consoleErrors.push(String(err)))

    const shotName = company.name.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/(^-|-$)/g, '')
    const shot = (label) =>
      page.screenshot({ path: path.join(SCRIPT_DIR, `verify-demo-${shotName}-${label}.png`) })

    // ---------- register (or log in, if a previous run already created this tenant) ----------
    await page.goto(`${BASE_URL}/register`, { waitUntil: 'networkidle0' })
    await clearAndType(page, 'input[autocomplete="organization"]', company.name)
    await clearAndType(page, 'input[autocomplete="username"]', company.email)
    await clearAndType(page, 'input[autocomplete="new-password"]', PASSWORD)
    await page.click('button[type="submit"]')
    await page.waitForFunction(
      () =>
        location.pathname === '/welcome' ||
        [...document.querySelectorAll('.field-error')].some((el) => el.textContent.includes('already registered')),
      { timeout: 10000 },
    )

    const alreadyRegisteredError = await page
      .$$eval('.field-error', (els) => els.map((el) => el.textContent).find((t) => t && t.includes('already registered')))
      .catch(() => null)
    if (alreadyRegisteredError) {
      console.log('[REGISTER] Tenant already exists from a previous run - logging in instead.')
      await page.goto(`${BASE_URL}/login`, { waitUntil: 'networkidle0' })
      await clearAndType(page, 'input[type="email"]', company.email)
      await clearAndType(page, 'input[type="password"]', PASSWORD)
      await Promise.all([
        page.click('button[type="submit"]'),
        page.waitForSelector('.dashboard-header', { timeout: 10000 }),
      ])
      // In-app nav click, NOT page.goto() - the auth token lives only in
      // React state (see AuthContext.jsx), so a real browser navigation
      // would immediately log this session back out.
      await Promise.all([
        page.click('.dashboard-nav a[href="/data-onboarding"]'),
        page.waitForFunction(() => location.pathname === '/data-onboarding', { timeout: 10000 }),
      ])
    } else {
      await page.waitForSelector('.register-welcome', { timeout: 10000 })
      const welcomeHeading = await page.$eval('.register-welcome h1', (el) => el.textContent)
      console.log('[REGISTER] Registered fresh tenant. Welcome heading:', welcomeHeading)
      await page.click('.register-welcome-button-primary')
      await page.waitForFunction(() => location.pathname === '/data-onboarding', { timeout: 10000 })
    }
    await page.waitForSelector('.onboarding-progress', { timeout: 10000 })

    // A re-run against a tenant from a previous pass resumes wherever that
    // run left off (mapping or report step, per GET onboarding-status) -
    // "Start Over" / "Back" both return to a clean step-1 upload, which is
    // what a real re-verification of the upload endpoint needs.
    const resumedStep = await page.$eval(
      '.onboarding-progress-active .onboarding-progress-label',
      (el) => el.textContent,
    )
    if (resumedStep !== '1. Upload') {
      console.log(`[RESUME] Landed on "${resumedStep}" from a previous run - starting over for a clean re-verification.`)
      const clicked = await page.evaluate(() => {
        const buttons = [...document.querySelectorAll('.onboarding-actions button')]
        const target = buttons.find((b) => b.textContent.trim() === 'Start Over') || buttons[0] // "Back" on the mapping step
        target?.click()
        return Boolean(target)
      })
      if (!clicked) throw new Error('No button found to return to the upload step from a resumed session.')
      await page.waitForSelector('input[type="file"]', { timeout: 10000 })
    }

    // ---------- upload (the real endpoint - this is what may reject Redline) ----------
    const csvPath = path.join(SAMPLES_DIR, company.csv)
    if (!fs.existsSync(csvPath)) throw new Error(`Sample CSV not found: ${csvPath}`)
    const fileInput = await page.$('input[type="file"]')
    await fileInput.uploadFile(csvPath)

    if (company.expectRejected) {
      await page.waitForSelector('.upload-error', { timeout: 15000 })
      const rejectionText = await page.$eval('.upload-error', (el) => el.textContent.trim())
      console.log('\n[UPLOAD REJECTED] Real response from POST /api/onboarding/upload:')
      console.log('  ', rejectionText)
      const stillOnUploadStep = await page.$eval(
        '.onboarding-progress-active .onboarding-progress-label',
        (el) => el.textContent,
      )
      console.log('[UPLOAD REJECTED] Wizard step after rejection (should still be Upload):', stillOnUploadStep)
      await shot('upload-rejected')
      results.push({ company: company.name, outcome: 'rejected', rejectionText })
      await page.close()
      continue
    }

    // ---------- mapping ----------
    await page.waitForSelector('.mapping-table', { timeout: 15000 })
    const suggestedRoles = await page.evaluate(() => {
      const rows = [...document.querySelectorAll('.mapping-table tbody tr')]
      return rows.map((r) => [r.children[0].textContent.trim(), r.querySelector('select').value])
    })
    console.log('[MAPPING] Auto-suggested roles:', JSON.stringify(Object.fromEntries(suggestedRoles)))
    await shot('mapping-before-fixes')

    for (const [column, role] of company.mappingFixes) {
      await setMappingRole(page, column, role)
    }
    if (company.mappingFixes.length) {
      const fixedRoles = await page.evaluate(() => {
        const rows = [...document.querySelectorAll('.mapping-table tbody tr')]
        return rows.map((r) => [r.children[0].textContent.trim(), r.querySelector('select').value])
      })
      console.log('[MAPPING] Roles after manual fixes:', JSON.stringify(Object.fromEntries(fixedRoles)))
    }
    await shot('mapping-after-fixes')

    // ---------- validate ----------
    await Promise.all([
      page.click('.onboarding-actions button:not([disabled]):last-child'),
      page.waitForSelector('.verdict-banner', { timeout: 15000 }),
    ])

    const verdictText = await page.$eval('.verdict-banner', (el) => el.textContent.trim())
    const checks = await page.$$eval('.validation-check', (els) => els.map((el) => el.textContent.trim()))
    const sufficiencyHeading = await page.$eval('.data-sufficiency-card h3', (el) => el.textContent.trim())
    const sufficiencyStats = await page.$eval('.data-sufficiency-stats', (el) => el.textContent.trim())
    const guidance = await page
      .$eval('.data-sufficiency-guidance', (el) => el.textContent.trim())
      .catch(() => null)

    console.log('\n[VALIDATION REPORT]')
    console.log('  Verdict:', verdictText)
    console.log('  Checks:')
    for (const c of checks) console.log('    -', c)
    console.log('  Data sufficiency:', sufficiencyHeading)
    console.log('  Stats:', sufficiencyStats.replace(/\s+/g, ' '))
    console.log('  Guidance:', guidance || '(none - no warning)')
    await shot('validation-report')

    results.push({
      company: company.name,
      outcome: 'validated',
      verdictText,
      checks,
      sufficiencyHeading,
      sufficiencyStats: sufficiencyStats.replace(/\s+/g, ' '),
      guidance,
    })

    console.log('\n[CONSOLE ERRORS]:', consoleErrors.length ? consoleErrors : 'none')
    await page.close()
  }

  console.log(`\n${'='.repeat(70)}\nSUMMARY\n${'='.repeat(70)}`)
  console.log(JSON.stringify(results, null, 2))
} finally {
  await browser.close()
}
