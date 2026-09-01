import { Link } from 'react-router-dom'

// PLACEHOLDER CONTENT - see PrivacyPolicy.jsx's module comment; the same
// caveat applies here. Not real, in-effect terms.
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
            <p>[Describe acceptable use, account responsibilities, etc. here.]</p>
          </section>
          <section>
            <h2 style={{ fontSize: 17 }}>Billing</h2>
            <p>[Describe real billing/subscription terms here, if applicable.]</p>
          </section>
          <section>
            <h2 style={{ fontSize: 17 }}>Liability</h2>
            <p>[Real liability/warranty language, reviewed by counsel, goes here.]</p>
          </section>
          <section>
            <h2 style={{ fontSize: 17 }}>Contact</h2>
            <p>[Real contact address/email for legal questions goes here.]</p>
          </section>
        </div>
        <Link className="landing-signin-button" to="/" style={{ marginTop: 8 }}>
          Back
        </Link>
      </div>
    </div>
  )
}
