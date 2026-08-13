# Pipeline figure: replace Mermaid with a hand-authored SVG

Date: 2026-08-13
Status: approved

## Problem

The `## The pipeline` section of the README presents the project pipeline as a
Mermaid `flowchart LR`. On GitHub that renders as a shrink-to-fit graphic with a
pan/zoom button overlay: the nine-node horizontal chain is scaled down until
the node text is tiny, and looking around requires clicking the viewer
controls. Both the viewer UI and the small text are unwanted.

## Decision

Replace the Mermaid block with a static, hand-authored SVG committed at
`figures/pipeline.svg` and embedded with a plain image tag. A plain `<img>`
embed never gets GitHub's diagram viewer chrome, and a hand-authored SVG gives
full control over layout and type size. Alternatives considered and rejected:

- ASCII diagram in a code block — no assets and full code-font size, but less
  polished; the README is a portfolio centerpiece.
- Numbered markdown list — biggest text, but loses the boxes-and-arrows shape.
- Generated SVG (matplotlib or graphviz script) — regenerable, but clunkier
  layout/typography than hand-authoring, and graphviz would add a system
  dependency.

## Design

**README change.** The Mermaid fence under `## The pipeline` is replaced by
`![The pipeline: from Gaussians to a tuned SVM](figures/pipeline.svg)`. The
explanatory prose paragraph below it stays unchanged.

**The asset.** `figures/pipeline.svg`, hand-authored:

- Nine stages flow in three serpentine rows sized to GitHub's README column
  (~830 px wide), so each box gets large type (~16–18 px) instead of
  shrink-to-fit text. Edge labels — "condition on data", "maximize LML
  (L-BFGS-B)" — sit on the arrows they belong to; "vs random search" and
  "propose → evaluate → refit" annotate the final stages.
- White background baked in with dark text, matching the repo's existing
  white-background PNG figures, so it reads in both GitHub themes.
- System font stack (`-apple-system, Segoe UI, Helvetica, Arial`) declared in
  the SVG; no embedded font files.

**Editing later.** Wording changes are edits to `<text>` elements in the file;
layout changes may need coordinate adjustments by hand (accepted trade-off).

## Verification

Render the SVG in a browser at README column width and screenshot it to check
layout, text size, and arrow placement. After push, a look at GitHub confirms
the embed has no zoom/pan chrome.
