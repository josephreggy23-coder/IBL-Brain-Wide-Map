# Decision Geometry Explorer

Interactive companion to the Decision Geometry Python analysis. The interface
uses exact arrays generated from the pinned IBL Brain Wide Map session and
provides synchronized time-resolved decoding, cross-temporal stability, and
regional decoding views.

## Development

```powershell
npm install
npm run dev
```

The explorer data is generated from the repository root:

```powershell
.\.venv\Scripts\python scripts\export_web_payload.py
```

Run the site checks with `npm test`. The production build is emitted to
`web/dist/` for Sites deployment.
