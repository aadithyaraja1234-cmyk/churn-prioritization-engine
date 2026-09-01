import { Link } from 'react-router-dom'

// PLACEHOLDER CONTENT. This exists so the /privacy route/link isn't a dead
// 404 and the registration flow has something real to link to, but the
// actual practices described below (what's collected, retention, etc.) need
// to be filled in with this product's real data-handling details - and
// reviewed by someone qualified to sign off on privacy/legal copy - before
// this is presented as this company's real policy.
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
            <p>[Describe what customer/account data this product actually collects here.]</p>
          </section>
          <section>
            <h2 style={{ fontSize: 17 }}>How we use it</h2>
            <p>[Describe the real purposes data is used for here.]</p>
          </section>
          <section>
            <h2 style={{ fontSize: 17 }}>Data retention</h2>
            <p>[Describe how long data is kept and how it can be deleted here.]</p>
          </section>
          <section>
            <h2 style={{ fontSize: 17 }}>Contact</h2>
            <p>[Real contact address/email for privacy requests goes here.]</p>
          </section>
        </div>
        <Link className="landing-signin-button" to="/" style={{ marginTop: 8 }}>
          Back
        </Link>
      </div>
    </div>
  )
}
