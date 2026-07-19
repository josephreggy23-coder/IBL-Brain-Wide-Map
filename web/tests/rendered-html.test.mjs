import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request("http://localhost/", {
      headers: { accept: "text/html", host: "decision-geometry.test" },
    }),
    { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } },
    { waitUntil() {}, passThroughOnException() {} },
  );
}

test("server-renders the Decision Geometry explorer", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>Decision Geometry \| IBL Neuropixels Explorer<\/title>/i);
  assert.match(html, /Neural population dynamics of a real decision/);
  assert.match(html, /Population information through time/);
  assert.match(html, /Peak choice readout/);
  assert.match(html, /82\.8/);
  assert.match(html, /Time-resolved neural decoding/);
  assert.match(html, /DANDI 000409/);
  assert.match(html, /<meta property="og:image" content="https:\/\/decision-geometry\.test\/og\.png"/i);
  assert.doesNotMatch(html, /codex-preview|SkeletonPreview|Your site is taking shape/);
});

test("ships exact analysis data and visual assets", async () => {
  const [payloadText, page, layout] = await Promise.all([
    readFile(new URL("../app/analysis-data.json", import.meta.url), "utf8"),
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/layout.tsx", import.meta.url), "utf8"),
    access(new URL("../public/decision-geometry.png", import.meta.url)),
    access(new URL("../public/og.png", import.meta.url)),
  ]);
  const payload = JSON.parse(payloadText);

  assert.equal(payload.summary.trials, 541);
  assert.equal(payload.summary.units, 93);
  assert.equal(payload.time.length, 30);
  assert.equal(payload.crossTemporalChoice.length, 30);
  assert.ok(Object.keys(payload.regionDecoding).length >= 5);
  assert.match(page, /type Panel = "decoding" \| "stability" \| "regions"/);
  assert.match(page, /id="time-scrubber"/);
  assert.match(layout, /summary_large_image/);
});
