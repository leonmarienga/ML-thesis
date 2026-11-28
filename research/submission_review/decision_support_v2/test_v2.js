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

const fixture = payload.veto_fixture;
const evidence = payload.evidence;
const { "veto-primary": primary, "veto-other": other, "veto-regime": regime,
        "veto-rows": table, "interval-level": level, "interval-rows": intervalRows } = elements;
assert.equal(elements["track-a-score"].textContent, "830/912 (91.01%)");
assert.equal(elements["track-a-mae"].textContent, "$5.015M");
assert.equal(elements["archived-declaration"].textContent, "$1,243,478.26");
assert.equal(elements["historical-gates"].children.length, 4);
assert.match(page, /actual obligations below \$50 million/);
assert.match(page, /not an unbiased out-of-sample estimate/);
assert.match(page, /not actual human participants/);
assert.match(page, /Closed zones stay in the scored denominators/);
assert.match(elements["interval-message"].textContent, /281 of 308.*0 of 9/);
assert.match(elements["veto-count"].textContent, /45\/48.*34\/48/);
const contents = element => element.children.map(row=>row.children.map(c=>c.textContent));
assert.deepEqual(contents(intervalRows).map(r=>r.slice(1)), [
  ["281 / 308", "91.23%"], ["281 / 299", "93.98%"], ["0 / 9", "0.00%"]
]);
for (const selection of ["80","90"]) {
  level.change(selection);
  const rows = contents(intervalRows);
  for (const [i,key] of ["all","below","tail"].entries()) {
    const [covered,n] = evidence.intervals[selection][key];
    assert.equal(rows[i][1],covered+" / "+n);
    assert.equal(rows[i][2],(100*covered/n).toFixed(2)+"%");
  }
}
const arms = Object.keys(fixture.groups.all);
let n = 0;
for (const p of arms) {
  primary.change(p);
  assert.equal(other.children.length, 2);
  assert.notEqual(other.value,p);
  assert.ok(other.children.every(node=>node.value!==p));
  for (const b of arms.filter(a=>a!==p)) {
    other.change(b);
    for (const g of Object.keys(fixture.groups)) {
      regime.change(g);
      assert.match(elements["veto-caption"].textContent,new RegExp(fixture.counts[g]+" matched simulated worlds"));
      const a = fixture.groups[g][p].scores, c = fixture.groups[g][b].scores;
      for (const [i,m] of ["overall","high_vulnerability","least_served"].entries()) {
        const [,left,right,difference,range] = contents(table)[i];
        assert.equal(left,(100*a[m]).toFixed(2)+"%");
        assert.equal(right,(100*c[m]).toFixed(2)+"%");
        assert.ok(Math.abs(Number(difference)-100*(a[m]-c[m])) < .00500001);
        if (g==="all") {
          const parts = range.split(" to ").map(Number);
          for (const [i,value] of fixture.intervals[p+" "+b][m].entries()) assert.ok(Math.abs(parts[i] - value*100) < .00500001);
        } else {
          assert.equal(range,"Not estimated");
        }
      }
      n++;
    }
  }
}
assert.equal(n,24);
assert.ok(Object.values(fixture.groups).every(g=>Object.keys(g).length===3));
assert.ok(!raw.includes("stage1_action_sha256") && !raw.includes("scenario_id"));
console.log("PASS: two interval levels and all 24 distinct-arm/regime combinations; role labels, displayed values, interval reversal, warnings, aggregate-only fixture and sources.");
