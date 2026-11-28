(function () {
  "use strict";
  const state = JSON.parse(document.getElementById("data").textContent);
  const {evidence, veto_fixture: fixture} = state;
  const id = key => document.getElementById(key);
  const link = "https://github.com/leonmarienga/ML-thesis/blob/"+state.commit+"/";
  const titles = {
    scripted_human_rule: "Scripted proportional rule",
    optimizer_only: "Linear optimization",
    optimizer_with_scripted_veto: "Optimizer with scripted veto"
  };
  const roads = {all:"All road conditions",normal:"Normal",delay:"Delayed",closure:"Closure"};
  const metrics = {
    overall:"Overall served",
    high_vulnerability:"High-vulnerability served",
    least_served:"Least-served zone"
  };
  const percentage = x => (100*x).toFixed(2)+"%";
  const pp = x => {
    const n = Number((100*x).toFixed(2));
    return (n > 0 ? "+" : "")+n.toFixed(2);
  };
  const dollars = x => new Intl.NumberFormat("en-US",{style:"currency",currency:"USD"}).format(x);
  function add(parent, tag, value) {
    const item = document.createElement(tag);
    item.textContent = value;
    parent.appendChild(item);
    return item;
  }
  function row(parent, label, values) {
    const tr = add(parent,"tr","");
    add(tr,"th",label).scope = "row";
    for (const value of values) add(tr,"td",value);
  }
  function option(parent, value, label) { add(parent,"option",label).value = value; }

  const t = evidence.track_a;
  id("track-a-score").textContent = t.correct+"/"+t.n+" ("+percentage(t.correct/t.n)+")";
  id("track-a-mae").textContent = "$"+(t.mae.track_a_reported_historical_50_200m_overlay/1e6).toFixed(3)+"M";
  id("archived-declaration").textContent = dollars(evidence.example.amount);
  id("archived-context").textContent = "Example: DR-"+evidence.example.id+" • FY"+evidence.example.year+" • "+evidence.example.hazard+", "+evidence.example.state+". The listed amount is the archived FEMA Mission Assignment obligation, not the cost of damage or the amount of assistance delivered.";
  for (const [label,path] of Object.entries(t.gate_paths)) {
    const li = add(id("historical-gates"),"li","");
    add(li,"a",label).href = link+path;
  }
  id("track-link").href = link+"research/results/track_a/model_metrics.csv";
  id("interval-link").href = link+"research/results/uncertainty/interval_metrics.csv";

  function renderIntervals() {
    const level = id("interval-level").value;
    const data = evidence.intervals[level];
    const body = id("interval-rows");
    body.replaceChildren();
    const labels = {all:"All evaluated cases",below:"Below $50 million",tail:"$50 million and above"};
    for (const [key,label] of Object.entries(labels)) {
      const [covered,n] = data[key];
      row(body,label,[covered+" / "+n,percentage(covered/n)]);
    }
    id("interval-message").textContent = "At nominal "+level+"%, the interval covered "+data.all[0]+" of "+data.all[1]+" cases overall, but only "+data.tail[0]+" of "+data.tail[1]+" at or above $50 million.";
  }
  id("interval-level").value = "90";
  id("interval-level").addEventListener("change",renderIntervals);
  renderIntervals();

  const primary = id("veto-primary");
  const other = id("veto-other");
  const regime = id("veto-regime");
  Object.entries(titles).forEach(([key,label]) => option(primary,key,label));
  Object.entries(roads).forEach(([k,label]) => option(regime,k,label+" · "+fixture.counts[k]+" cases"));
  primary.value = "optimizer_with_scripted_veto";
  other.value = "optimizer_only";
  regime.value = "all";
  function comparators() {
    const previous = other.value;
    const choices = Object.keys(titles).filter(key=>key!==primary.value);
    other.replaceChildren();
    for (const key of choices) option(other,key,titles[key]);
    other.value = choices.includes(previous) ? previous : choices[0];
  }
  function renderVeto() {
    const p = primary.value;
    const c = other.value;
    const g = regime.value;
    const results = fixture.groups[g];
    const intervals = g==="all" ? fixture.intervals[p+" "+c] : null;
    const table = id("veto-rows");
    table.replaceChildren();
    for (const [key,label] of Object.entries(metrics)) {
      const a = results[p].scores[key];
      const b = results[c].scores[key];
      const range = intervals ? intervals[key].map(pp).join(" to ") : "Not estimated";
      row(table,label,[percentage(a),percentage(b),pp(a-b),range]);
    }
    const n = fixture.counts[g];
    const v = results.optimizer_with_scripted_veto.vetoes;
    id("veto-caption").textContent = roads[g]+" · "+n+" matched simulated worlds";
    id("veto-summary").textContent = titles[p]+" versus "+titles[c]+".";
    id("veto-count").textContent = "In this set the scripted veto fired "+v[0]+"/"+n+" times at dispatch one and "+v[1]+"/"+n+" at dispatch two.";
  }
  primary.addEventListener("change",()=>{ comparators();renderVeto(); });
  other.addEventListener("change",renderVeto);
  regime.addEventListener("change",renderVeto);
  comparators();
  renderVeto();
  const provenance = id("v2-provenance");
  add(provenance,"li","Pinned commit: "+state.commit);
  for (const [name,sha] of Object.entries(evidence.file_hashes)) {
    const li = add(provenance,"li","");
    add(li,"a",name).href = link+name;
    add(li,"code"," · SHA-256 "+sha);
  }
  for (const [name,sha] of Object.entries(fixture.file_hashes)) add(provenance,"li","Veto fixture "+name+" · SHA-256 "+sha);
}());
