/* Runs the complete embedded application, dispatching change events through a
   small interface mock. Run node test.js after python build.py. */
const fs = require("node:fs");
const vm = require("node:vm");
const assert = require("node:assert/strict");
const path = require("node:path");

class Element {
  constructor(tag) { this.tagName = tag; this.children = []; this.textContent = ""; this.listeners = {}; this.attrs = {}; }
  appendChild(child) { child.parent = this; this.children.push(child); return child; }
  get lastElementChild() { return this.children.at(-1); }
  remove() { this.parent.children = this.parent.children.filter(c => c !== this); }
  setAttribute(k, v) { this.attrs[k] = v; }
  addEventListener(event, fn) { this.listeners[event] = fn; }
  replaceChildren() { this.children = []; }
  change(value) { this.value = value; this.listeners.change(); }
}

const page = fs.readFileSync(path.join(__dirname, "index.html"), "utf-8");
const [,raw] = page.match(/<script type="application\/json" id="data">([\s\S]*?)<\/script>/);
const [,script] = page.match(/<script>([\s\S]*?)<\/script>/);
assert.match(page, /Research only/);
assert.match(page, /allocation panels contain simulations and no observed aid delivered or participant results/);
assert.doesNotMatch(page, /<(?:script|link)\b[^>]*(?:src|href)=/i);
const payload = JSON.parse(raw);
const elements = Object.fromEntries([...page.matchAll(/\bid="([^"]+)"/g)].map(([, id])=>[id,new Element(id)]));
elements.data.textContent = raw;
for (const tag of ["title", "desc"]) elements.chart.appendChild(new Element(tag));
vm.runInNewContext(script, {document:{
  getElementById:id=>elements[id],
  createElement:tag=>new Element(tag),
  createElementNS:(_,tag)=>new Element(tag)
}}, {timeout:2000});

const pick = elements.selection;
const table = elements.comparison;
const shown = elements["coverage-caption"];
assert.equal(elements.preference.children.length, 4);
assert.equal(elements.comparator.children.length, 89);
assert.equal(elements.regime.children.length, 4);
assert.equal(elements.hashes.children.length, 5);
assert.equal(elements.chart.children.filter(c=>c.tagName==="circle").length, 91);
assert.equal(pick.textContent, "Vulnerability · 60% floor versus Efficiency · 0% floor");
assert.deepEqual(table.children.map(row=>row.children[1].textContent), ["47.17%","68.77%","22.42%"]);
const tested = new Set();
const fmt = x => (x*100).toFixed(2)+"%";
for (const p of payload.preference_choices) {
  elements.preference.change(p.preference);
  for (const [reg, n] of Object.entries({all:48,...payload.counts})) {
    elements.regime.change(reg);
    for (const c of payload.pareto_points) {
      elements.comparator.change(c.policy+":"+c.floor);
      const inContext = x => reg==="all"?x:payload.disruption_sensitivity.find(r=>r.regime===reg&&r.policy===x.policy&&r.floor===x.floor);
      const left = inContext(p), right = inContext(c);
      assert.equal(pick.textContent.includes("undefined"), false);
      assert.match(shown.textContent, new RegExp(n+" simulated cases"));
      for (const [i, key] of ["overall","high_vulnerability","least_served"].entries()) {
        const [,a,b,d] = table.children[i].children.map(e=>e.textContent);
        assert.equal(a,fmt(left[key]));
        assert.equal(b,fmt(right[key]));
        assert.ok(Math.abs(Number(d)-100*(left[key]-right[key])) <= .005000000001);
      }
      tested.add(p.preference+"/"+c.policy+":"+c.floor+"/"+reg);
    }
  }
}
assert.equal(tested.size, 1424);
assert.equal(elements.chart.children[0].tagName, "title");
assert.equal(elements.chart.children[1].tagName, "desc");
assert.equal(elements.chart.children.filter(c=>c.tagName==="circle").length, 91);
console.log("PASS: 1,424 live JS preference/comparator/regime combinations; values, gaps, plot redraw, source and evidence labels.");
