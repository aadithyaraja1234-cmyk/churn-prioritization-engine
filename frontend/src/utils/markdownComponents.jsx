// Shared react-markdown component overrides, used by both CopilotChat and
// MorningBrief so a rendered table gets the same horizontal-scroll wrapper
// (the app-wide .table-wrap class) in either place.
export const MARKDOWN_COMPONENTS = {
  table: ({ children }) => (
    <div className="table-wrap">
      <table>{children}</table>
    </div>
  ),
}
