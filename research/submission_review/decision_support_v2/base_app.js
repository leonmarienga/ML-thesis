(function () {
  "use strict";
  const data = JSON.parse(document.getElementById("data").textContent);
  const byId = id => document.getElementById(id);
  const points = data.pareto_points;
  const keys = ["overall", "high_vulnerability", "least_served"];
  const labels = ["Overall service", "High-vulnerability service", "Least-served zone"];
  const names = {
    efficiency: "Efficiency", vulnerability: "Vulnerability", geography: "Geographic priority",
    mixed: "Mixed priorities", maxmin: "Max-min", expected: "Expected recourse",
    worst_branch: "Worst-branch recourse", cvar: "CVaR recourse", proportional: "Proportional"
  };
  const prefLabels = {
    "total service": "Total service", "vulnerability priority": "Vulnerability priority",
    "minimum coverage priority": "Minimum coverage priority", "balanced": "Balanced"
  };
  const regLabels = {all: "All road conditions", normal: "Normal", delay: "Delayed", closure: "Closure"};
  const pct = x => (100 * x).toFixed(2) + "%";
  const delta = x => {
    const rounded = Number((100 * x).toFixed(2));
    return (rounded > 0 ? "+" : "") + rounded.toFixed(2);
  };
  const pointKey = p => p.policy + ":" + p.floor;
  const caption = p => names[p.policy] + " · " + Math.round(100 * p.floor) + "% floor";
  const add = (parent, tag, value) => {
    const node = document.createElement(tag);
    node.textContent = value;
    parent.appendChild(node);
    return node;
  };
  const option = (id, value, label) => {
    add(byId(id), "option", label).value = value;
  };
  Object.entries(prefLabels).forEach(([key, label]) => option("preference", key, label));
  points.forEach(p => option("comparator", pointKey(p), caption(p)));
  Object.entries(regLabels).forEach(([key, label]) => {
    const count = key === "all" ? 48 : data.counts[key];
    option("regime", key, label + " · " + count + " cases");
  });
  byId("preference").value = "balanced";
  byId("comparator").value = "efficiency:0";
  byId("regime").value = "all";

  function draw(selected, comparator) {
    const svg = byId("chart");
    // Preserve the accessible title and description on redraw.
    while (svg.children.length > 2) svg.lastElementChild.remove();
    function mark(tag, attrs, label) {
      const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
      Object.entries(attrs).forEach(([name, value]) => el.setAttribute(name, value));
      if (label !== undefined) el.textContent = label;
      svg.appendChild(el);
      return el;
    }
    const x = p => 56 + (p - 0.30) * 2560;
    const y = p => 248 - p * 460;
    for (const tick of [.3, .35, .4, .45, .5]) {
      mark("line", {x1:x(tick),y1:18,x2:x(tick),y2:248,stroke:"#dfe9e4"});
      mark("text", {x:x(tick),y:270,"text-anchor":"middle","font-size":"13",fill:"#335055"}, Math.round(tick*100)+"%");
    }
    for (const tick of [0, .1, .2, .3, .4, .5]) {
      mark("line", {x1:56,y1:y(tick),x2:568,y2:y(tick),stroke:"#dfe9e4"});
      mark("text", {x:48,y:y(tick)+4,"text-anchor":"end","font-size":"13",fill:"#335055"}, Math.round(tick*100)+"%");
    }
    mark("text", {x:312,y:293,"text-anchor":"middle","font-size":"15",fill:"#133731"}, "Overall service");
    mark("text", {x:15,y:133,transform:"rotate(-90 15 133)","text-anchor":"middle","font-size":"15",fill:"#133731"}, "Least-served zone");
    points.forEach(p => {
      const dot = mark("circle", {cx:x(p.overall),cy:y(p.least_served),r:p.nondominated?4:3,fill:p.nondominated?"#087061":"#819091",opacity:p.nondominated?".76":".5"});
      const title = document.createElementNS("http://www.w3.org/2000/svg", "title");
      title.textContent = caption(p) + " | " + keys.map(k=>pct(p[k])).join(" / ");
      dot.appendChild(title);
    });
    [[selected,"#131f20",8],[comparator,"#683992",12]].forEach(([p, color, radius]) => {
      mark("circle", {cx:x(p.overall),cy:y(p.least_served),r:radius,stroke:color,"stroke-width":2,fill:"none"});
    });
  }
  function render() {
    const pref = byId("preference").value;
    const regime = byId("regime").value;
    const selected = data.preference_choices.find(p => p.preference === pref);
    const comparator = points.find(p => pointKey(p) === byId("comparator").value);
    const count = regime === "all" ? 48 : data.counts[regime];
    function context(p) {
      return regime === "all" ? p : data.disruption_sensitivity.find(row => row.regime === regime && pointKey(row) === pointKey(p));
    }
    const p = context(selected);
    const c = context(comparator);
    byId("weights").textContent = labels.map((label, i) => label + " " + data.weights[pref][i] + "%").join(" · ") + ". The archived preference pick does not change when you filter road conditions.";
    byId("selection").textContent = caption(selected) + " versus " + caption(comparator);
    byId("coverage-caption").textContent = regLabels[regime] + " · " + count + " simulated cases";
    const body = byId("comparison");
    body.replaceChildren();
    keys.forEach((key, i) => {
      const row = add(body, "tr", "");
      add(row, "th", labels[i]).scope = "row";
      [pct(p[key]), pct(c[key]), delta(p[key] - c[key])].forEach(value => add(row, "td", value));
    });
    draw(points.find(p => pointKey(p) === pointKey(selected)), comparator);
  }

  const repo = "https://github.com/leonmarienga/ML-thesis/blob/" + data.commit;
  byId("source-link").href = repo + "/research/continuous/transport_pareto/REPORT.md";
  byId("protocol-link").href = repo + "/research/continuous/BATCH14_PROTOCOL.md";
  byId("source-commit").textContent = "Pinned Git commit: " + data.commit;
  Object.entries(data.hashes).forEach(([file, hash]) => add(byId("hashes"), "li", file + ": " + hash));
  ["preference","comparator","regime"].forEach(id => byId(id).addEventListener("change", render));
  render();
}());
