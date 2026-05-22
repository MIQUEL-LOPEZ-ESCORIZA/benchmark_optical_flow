#!/usr/bin/env python
"""Generate a static optical-flow comparison viewer."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Optical Flow Benchmark Viewer</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #111315;
      --panel: #1a1d21;
      --panel-2: #22262b;
      --text: #f2f5f8;
      --muted: #9ea8b3;
      --line: #343a42;
      --accent: #78d6c6;
      --warn: #ffc66d;
    }}
    * {{ box-sizing: border-box; }}
    html, body {{ height: 100%; }}
    body {{
      margin: 0;
      overflow: hidden;
      background: var(--bg);
      color: var(--text);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    .app {{
      height: 100vh;
      display: grid;
      grid-template-rows: auto 1fr;
    }}
    .toolbar {{
      display: grid;
      grid-template-columns: minmax(280px, 1fr) auto auto auto;
      gap: 12px;
      align-items: center;
      padding: 10px 14px;
      background: var(--panel);
      border-bottom: 1px solid var(--line);
    }}
    label {{
      display: grid;
      gap: 4px;
      color: var(--muted);
      font-size: 12px;
      line-height: 1.1;
    }}
    select, button {{
      height: 34px;
      border: 1px solid var(--line);
      background: var(--panel-2);
      color: var(--text);
      border-radius: 6px;
      padding: 0 10px;
      font: inherit;
    }}
    button {{ cursor: pointer; }}
    .models {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      justify-content: flex-end;
      max-width: 55vw;
    }}
    .model-toggle {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      height: 30px;
      padding: 0 9px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--panel-2);
      color: var(--text);
      font-size: 13px;
      white-space: nowrap;
    }}
    .model-toggle input {{ margin: 0; accent-color: var(--accent); }}
    .grid {{
      min-height: 0;
      padding: 10px;
      display: grid;
      gap: 10px;
      grid-template-columns: repeat(var(--cols), minmax(0, 1fr));
      grid-auto-rows: minmax(0, 1fr);
    }}
    .card {{
      min-width: 0;
      min-height: 0;
      display: grid;
      grid-template-rows: auto 1fr;
      background: #08090a;
      border: 1px solid var(--line);
      border-radius: 8px;
      overflow: hidden;
    }}
    .title {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
      min-width: 0;
      padding: 6px 8px;
      background: var(--panel);
      border-bottom: 1px solid var(--line);
      font-size: 13px;
      line-height: 1.2;
    }}
    .title span:first-child {{
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }}
    .badge {{
      color: var(--muted);
      font-size: 11px;
      white-space: nowrap;
    }}
    video {{
      width: 100%;
      height: 100%;
      min-height: 0;
      display: block;
      object-fit: contain;
      background: #000;
    }}
    .missing {{
      display: grid;
      place-items: center;
      min-height: 0;
      color: var(--warn);
      font-size: 13px;
      padding: 16px;
      text-align: center;
    }}
    @media (max-width: 900px) {{
      .toolbar {{
        grid-template-columns: 1fr;
        align-items: stretch;
      }}
      .models {{ max-width: none; justify-content: flex-start; }}
    }}
  </style>
</head>
<body>
  <div class="app">
    <div class="toolbar">
      <label>
        Video
        <select id="videoSelect"></select>
      </label>
      <button id="playPause" type="button">Play/Pause</button>
      <button id="sync" type="button">Sync</button>
      <div id="modelToggles" class="models"></div>
    </div>
    <main id="grid" class="grid"></main>
  </div>

  <script>
    const MANIFEST = __MANIFEST__;
    const selectedModels = new Set(MANIFEST.models);

    const videoSelect = document.getElementById('videoSelect');
    const modelToggles = document.getElementById('modelToggles');
    const grid = document.getElementById('grid');
    const playPauseButton = document.getElementById('playPause');
    const syncButton = document.getElementById('sync');

    function labelFor(item) {{
      return `${{item.category}} / ${{item.stem}}`;
    }}

    function buildControls() {{
      MANIFEST.videos.forEach((item, index) => {{
        const option = document.createElement('option');
        option.value = item.key;
        option.textContent = labelFor(item);
        if (index === 0) option.selected = true;
        videoSelect.appendChild(option);
      }});

      MANIFEST.models.forEach((model) => {{
        const label = document.createElement('label');
        label.className = 'model-toggle';
        const input = document.createElement('input');
        input.type = 'checkbox';
        input.checked = true;
        input.addEventListener('change', () => {{
          if (input.checked) selectedModels.add(model);
          else selectedModels.delete(model);
          render();
        }});
        label.appendChild(input);
        label.appendChild(document.createTextNode(model));
        modelToggles.appendChild(label);
      }});
    }}

    function currentItem() {{
      return MANIFEST.videos.find((item) => item.key === videoSelect.value) || MANIFEST.videos[0];
    }}

    function card(title, badge, path) {{
      const node = document.createElement('section');
      node.className = 'card';
      const head = document.createElement('div');
      head.className = 'title';
      const t = document.createElement('span');
      t.textContent = title;
      const b = document.createElement('span');
      b.className = 'badge';
      b.textContent = badge || '';
      head.appendChild(t);
      head.appendChild(b);
      node.appendChild(head);

      if (path) {{
        const video = document.createElement('video');
        video.src = path;
        video.controls = true;
        video.loop = true;
        video.muted = true;
        video.preload = 'metadata';
        node.appendChild(video);
      }} else {{
        const missing = document.createElement('div');
        missing.className = 'missing';
        missing.textContent = 'No output video found';
        node.appendChild(missing);
      }}
      return node;
    }}

    function bestColumns(count) {{
      if (count <= 1) return 1;
      if (count <= 4) return 2;
      if (count <= 6) return 3;
      if (count <= 9) return 3;
      return 4;
    }}

    function render() {{
      const item = currentItem();
      const chosen = MANIFEST.models.filter((model) => selectedModels.has(model));
      const entries = [
        ['RGB input', 'source', item.rgb || item.original],
        ...chosen.map((model) => [model, 'flow', item.flows[model] || null]),
      ];
      grid.style.setProperty('--cols', bestColumns(entries.length));
      grid.replaceChildren(...entries.map(([title, badge, path]) => card(title, badge, path)));
    }}

    function videos() {{
      return Array.from(grid.querySelectorAll('video'));
    }}

    function syncVideos() {{
      const all = videos();
      if (!all.length) return;
      const t = all[0].currentTime || 0;
      all.forEach((video) => {{
        if (Math.abs(video.currentTime - t) > 0.08) video.currentTime = t;
      }});
    }}

    playPauseButton.addEventListener('click', async () => {{
      const all = videos();
      const shouldPlay = all.some((video) => video.paused);
      if (shouldPlay) {{
        syncVideos();
        for (const video of all) await video.play().catch(() => {{}});
      }} else {{
        all.forEach((video) => video.pause());
      }}
    }});

    syncButton.addEventListener('click', syncVideos);
    videoSelect.addEventListener('change', render);

    buildControls();
    render();
  </script>
</body>
</html>
"""


def rel(path: Path, root: Path) -> str:
    return Path(os.path.relpath(path, root)).as_posix()


def build_manifest(results_dir: Path, dataset_dir: Path) -> dict:
    models = sorted(
        p.name
        for p in results_dir.iterdir()
        if p.is_dir() and not p.name.startswith("_")
    )
    videos_by_key: dict[str, dict] = {}

    for preprocessed in sorted((results_dir / "_preprocessed").glob("*/*.mp4")):
        category = preprocessed.parent.name
        stem = preprocessed.stem
        key = f"{category}/{stem}"
        original = dataset_dir / category / f"{stem}.mp4"
        videos_by_key[key] = {
            "key": key,
            "category": category,
            "stem": stem,
            "rgb": rel(preprocessed, results_dir),
            "original": rel(original, results_dir) if original.exists() else None,
            "flows": {},
        }

    for model in models:
        for flow_path in sorted((results_dir / model).glob("*/*/flow_viz.mp4")):
            category = flow_path.parent.parent.name
            stem = flow_path.parent.name
            key = f"{category}/{stem}"
            if key not in videos_by_key:
                original = dataset_dir / category / f"{stem}.mp4"
                videos_by_key[key] = {
                    "key": key,
                    "category": category,
                    "stem": stem,
                    "rgb": rel(original, results_dir) if original.exists() else None,
                    "original": rel(original, results_dir) if original.exists() else None,
                    "flows": {},
                }
            videos_by_key[key]["flows"][model] = rel(flow_path, results_dir)

    return {
        "models": models,
        "videos": sorted(videos_by_key.values(), key=lambda x: (x["category"], x["stem"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir",
        default="/capstor/scratch/cscs/mlopezescoriza/dataset_benchmark_optical_flow/results",
    )
    parser.add_argument(
        "--dataset_dir",
        default="/capstor/scratch/cscs/mlopezescoriza/dataset_benchmark_optical_flow",
    )
    parser.add_argument("--output_name", default="viewer.html")
    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()
    dataset_dir = Path(args.dataset_dir).resolve()
    manifest = build_manifest(results_dir, dataset_dir)
    html = HTML_TEMPLATE.replace("__MANIFEST__", json.dumps(manifest, indent=2))
    output_path = results_dir / args.output_name
    output_path.write_text(html)
    print(output_path)


if __name__ == "__main__":
    main()
