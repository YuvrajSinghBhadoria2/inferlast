# inferlast architecture diagram

Generated with the **Archify** agent skill (tt-a1i/archify, v2.17, MIT) to
provide a polished, interactive, self-contained architecture diagram for the
inferlast launch posts.

## Files

- `inferlast.architecture.json` — the typed Archify source spec (the editable
  authoring input; `schema_version: 1`, `diagram_type: architecture`,
  `quality_profile: showcase`).
- `inferlast-architecture.html` — the delivered self-contained artifact
  (712 KB). Open in a browser for the interactive viewer: dark/light theme
  toggle, pan/zoom, search, focus, relationship tracing, **trace motion**
  (`meta.animation: "trace"` — finite edge-reveal animation on first
  view/presentation, respects `prefers-reduced-motion`), and the **Export**
  menu (PNG / SVG / WebM / 1200×630 share card). Canonical exports stay
  still; motion is for interactive/presentation viewing.
- `inferlast-architecture.visual-check.1440x900.light.png` — an automated
  browser screenshot at the 1440×900 desktop viewport (light theme) for quick
  reference.

## Provenance (measured evidence, not claims)

- **Validation (showcase):** `node bin/archify.mjs validate architecture
  <spec> --quality showcase --json` → 9/9 artifact checks pass,
  composition status `pass`, **0 errors, 0 warnings**.
- **Delivery:** `deliver` → checks 9/9 pass; spec SHA-256
  `7f5508c89...ab100b` (3752 B) → artifact SHA-256
  `ec2b904e4...897c0` (712089 B).
- **Browser evidence** (`visual-check`, headless Chrome):
  - 1440×900: no overflow, min projected node text 7.6 px
  - 1600×1000: no overflow, min 7.85 px
  - 1920×1080: no overflow, min 8.96 px
  - 2048×1320: no overflow, min 9.0 px
  - Containment status: `pass`; visual review: `pending` (perceptual check is
    a human/eyeball confirmation, not automated).

## Regenerate

```bash
# from the archived skill package:
cd ~/.agents/skills/archify
node bin/archify.mjs deliver architecture \
  docs/diagram/inferlast.architecture.json \
  docs/diagram/inferlast-architecture.html --quality showcase --json
```

## What it shows

Hub-and-spoke centered on `auto_optimizer` (the orchestrator): the CLI
(`run`/`audit`) drives it; the measurement probes `profiler`, `quantize`,
`decode` attach above it; the truth verifiers `trustcheck` and `gpucheck`
attach below it; and `auto_optimizer` emits the always-persisted **Proof
report**. The `gpucheck → profiler` data dependency (it consumes profiler's
measured overhead + latency) is recorded in the card text rather than as a
crossing edge, to keep the share-card clean.
