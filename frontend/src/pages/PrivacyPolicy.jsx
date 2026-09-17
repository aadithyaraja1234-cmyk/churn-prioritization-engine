import { Link } from 'react-router-dom'

// DEMO PLACEHOLDER CONTENT. This exists so the /privacy route/link isn't a
// dead 404 and the registration flow has something real to link to.
// Generic, non-binding filler written only so this page isn't empty during
// a live demo - NOT reviewed by counsel. The actual practices described
// below (what's collected, retention, etc.) need to be filled in with this
// product's real data-handling details - and reviewed by someone qualified
// to sign off on privacy/legal copy - before this is presented as this
// company's real policy.
export default function PrivacyPolicy() {
  return (
    <div className="landing-page">
      <div className="landing-card" style={{ maxWidth: 640, textAlign: 'left' }}>
        <h1 style={{ textAlign: 'center' }}>Privacy Policy</h1>
        <p className="not-available" style={{ alignSelf: 'stretch' }}>
          Placeholder — this page has not been reviewed or filled in with this product's real
          data-handling practices yet. Do not treat this as a real, in-effect policy.
        </p>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          <section>
            <h2 style={{ fontSize: 17 }}>What we collect</h2>
            <p>
              Account details you provide at registration (name, email, hashed password), and the
              customer-record CSV you upload for training. Uploaded data is stored per-tenant and is
              not shared across tenants.
            </p>
          </section>
          <section>
            <h2 style={{ fontSize: 17 }}>How we use it</h2>
            <p>Solely to run this demo: training and serving churn-prediction models for your account.</p>
          </section>
          <section>
            <h2 style={{ fontSize: 17 }}>Data retention</h2>
            <p>
              This is a demo environment — data may be reset or deleted at any time without notice.
              Don't upload real customer data you need retained.
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
