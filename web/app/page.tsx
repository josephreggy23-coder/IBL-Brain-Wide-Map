"use client";

import { useMemo, useState } from "react";
import data from "./analysis-data.json";

type Panel = "decoding" | "stability" | "regions";

const series = [
  { key: "stimulus", label: "Stimulus side", color: "#d95f02" },
  { key: "prior", label: "Block prior", color: "#13866b" },
  { key: "choice", label: "Wheel choice", color: "#0878b7" },
] as const;

const panels: Array<{ key: Panel; label: string }> = [
  { key: "decoding", label: "Information" },
  { key: "stability", label: "Code stability" },
  { key: "regions", label: "Regions" },
];

const width = 760;
const height = 360;
const margin = { top: 24, right: 22, bottom: 48, left: 58 };
const plotWidth = width - margin.left - margin.right;
const plotHeight = height - margin.top - margin.bottom;

function xScale(time: number) {
  return margin.left + ((time + 0.5) / 1.5) * plotWidth;
}

function yScale(value: number) {
  return margin.top + (1 - (value - 0.4) / 0.6) * plotHeight;
}

function linePath(values: number[]) {
  return values
    .map((value, index) => {
      const command = index === 0 ? "M" : "L";
      return `${command}${xScale(data.time[index]).toFixed(1)},${yScale(value).toFixed(1)}`;
    })
    .join(" ");
}

function heatColor(value: number, palette: "magma" | "viridis") {
  const t = Math.max(0, Math.min(1, (value - 0.45) / 0.4));
  if (palette === "magma") {
    const hue = 282 - t * 240;
    return `hsl(${hue} 72% ${18 + t * 55}%)`;
  }
  const hue = 264 - t * 210;
  return `hsl(${hue} ${48 + t * 25}% ${28 + t * 34}%)`;
}

function DecodingChart({ selectedIndex }: { selectedIndex: number }) {
  const ticks = [-0.5, 0, 0.5, 1];
  const yTicks = [0.5, 0.6, 0.7, 0.8, 0.9, 1];
  return (
    <svg className="chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-labelledby="decode-title decode-desc">
      <title id="decode-title">Time-resolved neural decoding</title>
      <desc id="decode-desc">Balanced accuracy for stimulus side, block prior, and wheel choice from 0.5 seconds before to 1 second after stimulus onset.</desc>
      {yTicks.map((tick) => (
        <g key={tick}>
          <line className="grid-line" x1={margin.left} x2={width - margin.right} y1={yScale(tick)} y2={yScale(tick)} />
          <text className="axis-label" x={margin.left - 12} y={yScale(tick) + 4} textAnchor="end">{tick.toFixed(1)}</text>
        </g>
      ))}
      {ticks.map((tick) => (
        <g key={tick}>
          <line className="tick-line" x1={xScale(tick)} x2={xScale(tick)} y1={height - margin.bottom} y2={height - margin.bottom + 6} />
          <text className="axis-label" x={xScale(tick)} y={height - 18} textAnchor="middle">{tick.toFixed(1)}</text>
        </g>
      ))}
      <line className="chance-line" x1={margin.left} x2={width - margin.right} y1={yScale(0.5)} y2={yScale(0.5)} />
      <line className="event-line" x1={xScale(0)} x2={xScale(0)} y1={margin.top} y2={height - margin.bottom} />
      <line className="selection-line" x1={xScale(data.time[selectedIndex])} x2={xScale(data.time[selectedIndex])} y1={margin.top} y2={height - margin.bottom} />
      {series.map((item) => (
        <g key={item.key}>
          <path className="data-line" d={linePath(data.decoding[item.key])} stroke={item.color} />
          <circle cx={xScale(data.time[selectedIndex])} cy={yScale(data.decoding[item.key][selectedIndex])} r="5" fill={item.color} stroke="#ffffff" strokeWidth="2" />
        </g>
      ))}
      <text className="axis-title" x={(margin.left + width - margin.right) / 2} y={height - 2} textAnchor="middle">Time from stimulus onset (s)</text>
      <text className="axis-title" transform={`translate(15 ${(margin.top + height - margin.bottom) / 2}) rotate(-90)`} textAnchor="middle">Balanced accuracy</text>
    </svg>
  );
}

function StabilityChart({ selectedIndex }: { selectedIndex: number }) {
  const matrix = data.crossTemporalChoice;
  const cellWidth = plotWidth / matrix.length;
  const cellHeight = plotHeight / matrix.length;
  return (
    <svg className="chart" viewBox={`0 0 ${width} ${height}`} role="img" aria-labelledby="stability-title stability-desc">
      <title id="stability-title">Cross-temporal choice decoding</title>
      <desc id="stability-desc">Balanced accuracy when a choice decoder trained at one time is tested at all other times.</desc>
      {matrix.map((row, rowIndex) =>
        row.map((value, columnIndex) => (
          <rect
            key={`${rowIndex}-${columnIndex}`}
            x={margin.left + columnIndex * cellWidth}
            y={margin.top + (matrix.length - 1 - rowIndex) * cellHeight}
            width={cellWidth + 0.4}
            height={cellHeight + 0.4}
            fill={heatColor(value, "magma")}
          />
        )),
      )}
      <rect className="matrix-selection" x={margin.left + selectedIndex * cellWidth} y={margin.top} width={cellWidth} height={plotHeight} />
      <rect className="matrix-selection" x={margin.left} y={margin.top + (matrix.length - 1 - selectedIndex) * cellHeight} width={plotWidth} height={cellHeight} />
      <line className="event-light" x1={xScale(0)} x2={xScale(0)} y1={margin.top} y2={height - margin.bottom} />
      <line className="event-light" x1={margin.left} x2={width - margin.right} y1={height - margin.bottom - ((0 + 0.5) / 1.5) * plotHeight} y2={height - margin.bottom - ((0 + 0.5) / 1.5) * plotHeight} />
      <text className="axis-title" x={(margin.left + width - margin.right) / 2} y={height - 8} textAnchor="middle">Test time</text>
      <text className="axis-title" transform={`translate(15 ${(margin.top + height - margin.bottom) / 2}) rotate(-90)`} textAnchor="middle">Train time</text>
      <text className="matrix-edge-label" x={margin.left} y={height - 28}>-0.5 s</text>
      <text className="matrix-edge-label" x={width - margin.right} y={height - 28} textAnchor="end">+1.0 s</text>
    </svg>
  );
}

function RegionChart({ selectedIndex }: { selectedIndex: number }) {
  const regions = Object.entries(data.regionDecoding).sort((a, b) => Math.max(...b[1]) - Math.max(...a[1]));
  const chartWidth = 880;
  const left = 320;
  const right = 20;
  const top = 28;
  const bottom = 48;
  const innerWidth = chartWidth - left - right;
  const innerHeight = height - top - bottom;
  const cellWidth = innerWidth / data.time.length;
  const cellHeight = innerHeight / regions.length;
  return (
    <svg className="chart region-chart" viewBox={`0 0 ${chartWidth} ${height}`} role="img" aria-labelledby="region-title region-desc">
      <title id="region-title">Choice information by anatomical region</title>
      <desc id="region-desc">Balanced choice-decoding accuracy over time for five regions with at least five retained units.</desc>
      {regions.map(([name, values], rowIndex) => (
        <g key={name}>
          <text className="region-label" x={left - 14} y={top + rowIndex * cellHeight + cellHeight / 2 + 4} textAnchor="end">{name}</text>
          {values.map((value, columnIndex) => (
            <rect
              key={columnIndex}
              x={left + columnIndex * cellWidth}
              y={top + rowIndex * cellHeight}
              width={cellWidth + 0.3}
              height={cellHeight + 0.3}
              fill={heatColor(value, "viridis")}
            />
          ))}
        </g>
      ))}
      <rect className="matrix-selection" x={left + selectedIndex * cellWidth} y={top} width={cellWidth} height={innerHeight} />
      <line className="event-light" x1={left + ((0 + 0.5) / 1.5) * innerWidth} x2={left + ((0 + 0.5) / 1.5) * innerWidth} y1={top} y2={height - bottom} />
      <text className="axis-title" x={left + innerWidth / 2} y={height - 8} textAnchor="middle">Time from stimulus onset</text>
      <text className="matrix-edge-label" x={left} y={height - 28}>-0.5 s</text>
      <text className="matrix-edge-label" x={chartWidth - right} y={height - 28} textAnchor="end">+1.0 s</text>
    </svg>
  );
}

export default function Home() {
  const peakIndex = data.decoding.choice.indexOf(Math.max(...data.decoding.choice));
  const [panel, setPanel] = useState<Panel>("decoding");
  const [selectedIndex, setSelectedIndex] = useState(peakIndex);
  const selectedTime = data.time[selectedIndex];
  const regionEntries = useMemo(() => Object.entries(data.regionDecoding), []);
  const strongestRegion = regionEntries.reduce((best, current) =>
    current[1][selectedIndex] > best[1][selectedIndex] ? current : best,
  );

  return (
    <main>
      <header className="topbar">
        <a className="brand" href="#top" aria-label="Decision Geometry home">
          <span className="brand-mark" aria-hidden="true"><span /><span /><span /></span>
          <span>Decision Geometry</span>
        </a>
        <nav className="topnav" aria-label="Primary navigation">
          <a href="#explorer">Explorer</a>
          <a href="#method">Method</a>
          <a href="#source">Data</a>
          <a className="github-link" href="https://github.com/josephreggy23-coder/IBL-Brain-Wide-Map">GitHub</a>
        </nav>
      </header>

      <section className="intro" id="top">
        <div>
          <p className="eyebrow">IBL Brain Wide Map / DANDI 000409</p>
          <h1>Neural population dynamics of a real decision.</h1>
          <p className="lede">Explore when stimulus evidence, learned prior, and wheel choice become readable from simultaneously recorded Neuropixels activity.</p>
        </div>
        <div className="session-stamp" aria-label="Session provenance">
          <span>SUBJECT</span><strong>NYU-39</strong>
          <span>SESSION</span><strong>6ed57216</strong>
          <span>WINDOW</span><strong>-0.5 to +1.0 s</strong>
        </div>
      </section>

      <section className="metric-grid" aria-label="Analysis summary">
        <article className="metric-card"><span>Valid trials</span><strong>{data.summary.trials}</strong><small>visual decisions</small></article>
        <article className="metric-card"><span>Retained units</span><strong>{data.summary.units}</strong><small>strict IBL quality filter</small></article>
        <article className="metric-card"><span>Regions</span><strong>{data.summary.regions}</strong><small>anatomical labels</small></article>
        <article className="metric-card accent-card"><span>Peak choice readout</span><strong>{(data.summary.peakChoiceAccuracy * 100).toFixed(1)}%</strong><small>at {Math.round(data.summary.peakChoiceTime * 1000)} ms</small></article>
      </section>

      <section className="explorer" id="explorer">
        <div className="section-heading">
          <div><p className="eyebrow">Interactive analysis</p><h2>Population information through time</h2></div>
          <div className="segmented" role="tablist" aria-label="Analysis view">
            {panels.map((item) => (
              <button key={item.key} type="button" role="tab" aria-selected={panel === item.key} onClick={() => setPanel(item.key)}>{item.label}</button>
            ))}
          </div>
        </div>

        <div className="analysis-layout">
          <div className="plot-area">
            {panel === "decoding" && <DecodingChart selectedIndex={selectedIndex} />}
            {panel === "stability" && <StabilityChart selectedIndex={selectedIndex} />}
            {panel === "regions" && <RegionChart selectedIndex={selectedIndex} />}
            <div className="time-control">
              <label htmlFor="time-scrubber">Selected time <strong>{selectedTime >= 0 ? "+" : ""}{selectedTime.toFixed(3)} s</strong></label>
              <input id="time-scrubber" type="range" min="0" max={data.time.length - 1} value={selectedIndex} onChange={(event) => setSelectedIndex(Number(event.target.value))} />
              <div><span>-0.5 s</span><span>stimulus onset</span><span>+1.0 s</span></div>
            </div>
          </div>

          <aside className="readout" aria-live="polite">
            <p className="readout-time">t = {selectedTime >= 0 ? "+" : ""}{selectedTime.toFixed(3)} s</p>
            {series.map((item) => (
              <div className="readout-row" key={item.key}>
                <span><i style={{ background: item.color }} />{item.label}</span>
                <strong>{(data.decoding[item.key][selectedIndex] * 100).toFixed(1)}%</strong>
              </div>
            ))}
            <div className="readout-rule" />
            <p className="readout-label">Strongest region now</p>
            <strong className="region-name">{strongestRegion[0].replace(/ \(n=\d+\)/, "")}</strong>
            <p className="readout-note">{(strongestRegion[1][selectedIndex] * 100).toFixed(1)}% balanced choice accuracy. Above 50% means a linear model reads choice information better than chance.</p>
          </aside>
        </div>
      </section>

      <section className="method" id="method">
        <div className="section-heading"><div><p className="eyebrow">Computation</p><h2>From remote spikes to interpretable dynamics</h2></div></div>
        <ol className="pipeline">
          <li><span>01</span><strong>Stream</strong><p>Read only required byte ranges from the pinned public NWB asset.</p></li>
          <li><span>02</span><strong>Filter</strong><p>Keep units passing all IBL quality metrics and presence thresholds.</p></li>
          <li><span>03</span><strong>Align</strong><p>Bin spikes in 50 ms windows around visual stimulus onset.</p></li>
          <li><span>04</span><strong>Decode</strong><p>Use stratified cross-validation for stimulus, prior, and choice.</p></li>
          <li><span>05</span><strong>Compare</strong><p>Measure low-dimensional trajectories, temporal stability, and regions.</p></li>
        </ol>
      </section>

      <section className="publication-view">
        <div className="section-heading"><div><p className="eyebrow">Publication export</p><h2>The complete analysis dashboard</h2></div><a href="/decision-geometry.png">Open full resolution</a></div>
        <img src="/decision-geometry.png" alt="Four-panel Decision Geometry scientific dashboard" />
      </section>

      <section className="source" id="source">
        <div><p className="eyebrow">Data provenance</p><h2>Public, pinned, and inspectable.</h2><p>The source is one published session from the International Brain Laboratory Brain Wide Map. The explorer uses exact derived values from the repository analysis; the raw recording remains on DANDI.</p></div>
        <dl>
          <div><dt>Dandiset</dt><dd><a href="https://dandiarchive.org/dandiset/000409">000409</a></dd></div>
          <div><dt>Version</dt><dd>0.260309.1324</dd></div>
          <div><dt>Asset</dt><dd>882e2ff6-1fde-4518-8797-5d5892379739</dd></div>
          <div><dt>Format</dt><dd>Neurodata Without Borders</dd></div>
        </dl>
      </section>

      <footer><span>Decision Geometry</span><p>Real neural data. Reproducible computation. Careful claims.</p><a href="https://github.com/josephreggy23-coder/IBL-Brain-Wide-Map">Source code</a></footer>
    </main>
  );
}
