# Autograd Playground

An interactive browser playground for visualizing forward passes, backpropagation, and computation graphs.

## What It Does

- Builds a computation graph from simple scalar expressions.
- Steps through forward and backward passes node by node.
- Shows chain-rule flow in the sidebar and compact gradient annotations on the graph.
- Supports zoom, pan, fit, and focus for large graphs.
- Exports the current graph as JSON or SVG.
- Exports a full timeline of graph states across all stages.
- Includes light and dark themes.

## Running It

This is a static app. Serve the repository root with any simple HTTP server and open `index.html`.

Example:

```bash
python3 -m http.server 4181
```

Then visit `http://127.0.0.1:4181/`.

## Controls

- `F`: forward step
- `B`: backward step
- `Space`: autoplay
- `R`: reset
- Mouse wheel: zoom graph
- Drag on canvas: pan graph
- Double-click graph: fit view

## Export Options

- `Current JSON`: current state of the graph
- `All Stages`: full forward/backward timeline export
- `SVG`: current visual graph export

Programmatic access is available from the browser console:

```js
window.AutogradPlayground.snapshot()
window.AutogradPlayground.timeline()
window.AutogradPlayground.saveJSON()
window.AutogradPlayground.saveStagesJSON()
window.AutogradPlayground.saveSVG()
window.AutogradPlayground.fitView()
window.AutogradPlayground.zoomIn()
window.AutogradPlayground.zoomOut()
window.AutogradPlayground.focusActive()
```

## Project Structure

- `index.html`: app shell and DOM structure
- `styles.css`: visual system, layout, and themes
- `js/app.js`: runtime, state, UI wiring, exports, camera controls
- `js/examples.js`: example catalog

## Large-Graph Readability

The app now handles dense graphs better, but the next upgrades that would make it even stronger are:

- Minimap overview with viewport box
- Auto-follow mode for the active node during stepping
- Semantic grouping/collapsing for repeated subexpressions
- Stage scrubber for fast timeline navigation
