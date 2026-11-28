"use strict";
// Render a locked thesis viewer exactly as shipped. Crop by native element bounds.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const crypto = require("node:crypto");
const path = require("node:path");
const {pathToFileURL} = require("node:url");
const {createRequire} = require("node:module");

const APP_SHA = "9ddf0fedf618e7e9e3df5a194890ecb1f88883d4a48a4917cba2d423f7e84203";
const BROWSER_SHA = "1f698044a68e18c1ce0cce74be25413f176806f19f15a44de5bb02cd578996af";
const sha = bytes => crypto.createHash("sha256").update(bytes).digest("hex");

async function main() {
  assert.equal(process.argv.length, 6,
    "Usage: node capture_f10.cjs PINNED_INDEX.html BROWSER_RUNTIME_DIR F10.png F10.json");
  const [html, runtime, output, recordPath] = process.argv.slice(2).map(p => path.resolve(p));
  assert.notEqual(output, recordPath);
  assert(!fs.existsSync(output) && !fs.existsSync(recordPath), "Use new output paths");
  assert.equal(sha(fs.readFileSync(html)), APP_SHA, "Unrecognized viewer bytes");
  const runtimeRequire = createRequire(path.join(runtime, "package.json"));
  const {chromium} = runtimeRequire("playwright");
  assert.equal(runtimeRequire("playwright/package.json").version, "1.62.1");
  const moduleEntry = runtimeRequire.resolve("@sparticuz/chromium");
  const modulePackage = path.resolve(path.dirname(moduleEntry), "../package.json");
  assert.equal(JSON.parse(fs.readFileSync(modulePackage)).version, "158.0.0");
  const browserPkg = runtimeRequire("@sparticuz/chromium").default;
  const exe = await browserPkg.executablePath();
  assert.equal(sha(fs.readFileSync(exe)), BROWSER_SHA, "Unrecognized Chromium binary");
  const viewport = {width:1000,height:900};
  const deviceScaleFactor = 2;
  const browser = await chromium.launch({executablePath:exe,args:browserPkg.args,headless:true});
  let png, report;
  try {
    assert.equal(browser.version(), "158.0.8127.0");
    const page = await browser.newPage({viewport,deviceScaleFactor,locale:"en-US"});
    const errors = [], requests = [];
    page.on("pageerror", e => errors.push(e.message));
    page.on("request", r => { if (/^https?:/.test(r.url())) requests.push(r.url()); });
    await page.route(/^https?:/,r=>r.abort());
    await page.goto(pathToFileURL(html).href,{waitUntil:"load"});
    await page.evaluate(() => document.fonts.ready);
    assert.match(await page.locator("main h1").innerText(),/Disaster relief decision support/);
    const crop = page.locator('section[aria-labelledby="veto"]');
    const roads = crop.locator("#veto-regime");
    const comparator = crop.locator("#veto-other");
    await comparator.focus();
    await page.keyboard.press("End");
    assert.equal(await comparator.inputValue(), "optimizer_only");
    assert.equal(await roads.inputValue(), "all");
    const before = await crop.locator("#veto-caption").innerText();
    assert.match(before, /48 matched simulated worlds/);
    const beforeIntervals = await crop.locator("#veto-rows tr td:last-child").allTextContents();
    assert(beforeIntervals.length === 3 && beforeIntervals.every(v => v.includes(" to ")));
    await roads.focus();
    await page.keyboard.press("End");
    assert.equal(await roads.inputValue(), "closure");
    const after = await crop.locator("#veto-caption").innerText();
    assert.equal(after, "Closure · 15 matched simulated worlds");
    assert.deepEqual(await crop.locator("#veto-rows tr td:last-child").allTextContents(),
                     Array(3).fill("Not estimated"));
    assert.equal(await crop.locator("#veto-primary").inputValue(),"optimizer_with_scripted_veto");
    assert.equal(await crop.locator("#veto-other").inputValue(),"optimizer_only");
    const table = await crop.locator("#veto-rows tr").evaluateAll(
      a=>a.map(row=>Array.from(row.cells, c=>c.innerText)));
    assert.deepEqual(table, [
      ["Overall served","38.54%","43.79%","-5.24","Not estimated"],
      ["High-vulnerability served","41.35%","63.06%","-21.71","Not estimated"],
      ["Least-served zone","13.11%","4.05%","+9.06","Not estimated"],
    ]);
    const bounds = await crop.boundingBox();
    assert(bounds.width > 850 && bounds.width <= viewport.width && bounds.height < viewport.height);
    png = await crop.screenshot({animations:"disabled",scale:"device"});
    assert.equal(png.subarray(1,4).toString(),"PNG");
    assert.deepEqual(errors,[]);
    assert.deepEqual(requests,[]);
    report = {
      artifact: {html_sha256:APP_SHA, chromium_executable_sha256:BROWSER_SHA,
                 chromium_version:browser.version(), playwright_version:"1.62.1", chromium_package:"158.0.0"},
      capture: {selector:'section[aria-labelledby="veto"]', viewport_css_px:viewport,
                device_scale_factor:deviceScaleFactor, box_css_px:{width:bounds.width,height:bounds.height},
                png_px:{width:png.readUInt32BE(16),height:png.readUInt32BE(20)}, png_sha256:sha(png),
                state_change:"Open file://; select Linear optimization in Comparison policy; focus Road conditions, press End", proof_of_change:{before,after},
                table, page_errors:errors, http_requests:requests},
      scope:"Single actual screenshot of offline synthetic comparison; no modeled real-world policy effect, participants, accessibility or usability claim."
    };
  } finally {
    await browser.close();
  }
  fs.writeFileSync(output,png,{flag:"wx"});
  fs.writeFileSync(recordPath,JSON.stringify(report,null,2)+"\n",{flag:"wx"});
  console.log(JSON.stringify(report.capture));
}
main().catch(err=>{console.error(err);process.exitCode=1;});
