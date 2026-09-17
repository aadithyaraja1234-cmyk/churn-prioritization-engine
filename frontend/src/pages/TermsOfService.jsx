import { Link } from 'react-router-dom'

// DEMO PLACEHOLDER CONTENT - see PrivacyPolicy.jsx's module comment; the
// same caveat applies here. Generic, non-binding filler written only so
// this page isn't a dead end during a live demo - NOT reviewed by counsel,
// NOT this product's real terms. Must be replaced with real, reviewed
// terms before any real account or real customer data touches this app.
export default function TermsOfService() {
  return (
    <div className="landing-page">
      <div className="landing-card" style={{ maxWidth: 640, textAlign: 'left' }}>
        <h1 style={{ textAlign: 'center' }}>Terms of Service</h1>
        <p className="not-available" style={{ alignSelf: 'stretch' }}>
          Placeholder — this page has not been reviewed or filled in with this product's real terms
          yet. Do not treat this as a real, in-effect agreement.
        </p>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <section>
            <h2 style={{ fontSize: 17 }}>Use of the service</h2>
            <p>
              This demo lets you register a tenant account, upload a CSV of customer records, and
              train churn-prediction models against it. Use it only with data you have the right to
              upload, and only for evaluation purposes. Don't attempt to disrupt the service, access
              another tenant's data, or upload malicious files.
            </p>
          </section>
          <section>
            <h2 style={{ fontSize: 17 }}>Billing</h2>
            <p>This is a demo environment — no payment is collected and no subscription is created.</p>
          </section>
          <section>
            <h2 style={{ fontSize: 17 }}>Liability</h2>
            <p>
              Provided "as is," for demonstration purposes, with no uptime or accuracy guarantee.
              Predictions and analytics are illustrative, not a basis for real business decisions.
            </p>
          </section>
          <section>
            <h2 style={{ fontSize: 17 }}>Contact</h2>
            <p>See this project's repository for how to reach the maintainers.</p>
          </section>
        </div>
        <Link className="landing-signin-button" to="/" style={{ marginTop: 8 }}>
          Back
        </Link>
      </div>
    </div>
  )
}
