# Constrained two-day safe-water allocation

The simulation uses cubic metres of clean water and explicitly models depot inventory and road throughput. It uses the WHO emergency reference of 15 L per person per day for hypothetical domestic demand, with stylized populations, available local supply, vulnerabilities, replenishment and roads. None of these scenario quantities represents a measured municipal deficit, a FEMA obligation or a cost forecast. The four future states vary replenishment, route capacity and demand together. [WHO water guidance](https://www.who.int/teams/environment-climate-change-and-health/water-sanitation-and-health/environmental-health-in-emergencies/humanitarian-emergencies), [scenario inputs](results/scenario_inputs.jsonl.gz).

The two-stage linear programs optimize vulnerability-weighted water deficits subject to no overdelivery, initial and replenished inventory caps, and day- and state-specific road caps. The morning shipment is fixed across all futures. Day-two recourse uses the state then realized. Each myopic and planning arm sees the same realized day-two data only after its morning policy has committed. The CVaR arm adds a penalty for the upper quartile of modeled loss. Day-one unmet need remains a loss, even if depletion shifts to day two. Policies passed all 960 feasibility checks on 240 paired scenarios. [Protocol](../BATCH3_PROTOCOL.md), [sample dispatch](results/example_dispatch.csv).

| Policy | Mean total service | Mean day-one service | Higher-vulnerability service | 10th-percentile weighted service |
| --- | ---: | ---: | ---: | ---: |
| Myopic fractional priority | 68.52% | 84.39% | 69.94% | 41.76% |
| Myopic vulnerability priority | 68.52% | 84.39% | 84.50% | 42.84% |
| Expected-loss recourse | 68.45% | 83.92% | 84.81% | 42.84% |
| CVaR recourse | 67.89% | 80.66% | 85.76% | 43.73% |

CVaR raised bottom-decile weighted service by **0.89 percentage points** (scenario-paired interval 0.00 to +1.07), short of the registered two-point gain. It raised higher-vulnerability service by **1.26 points** (+0.89 to +1.68) and lowered mean total service by **0.63 points** (−0.90 to −0.38), including an average 3.74-point reduction in day-one service. Expected-loss recourse made little difference to the lower tail. Vulnerability priority in the simple myopic policy allocated substantially more service to the higher-vulnerability half for essentially the same total volume as the fractional-priority diagnostic. This difference follows from the specified objectives and constraints. It is a simulation result, and neither recourse candidate met the integration gate. [Policy results](results/policy_metrics.csv), [paired uncertainty](results/paired_scenario_intervals.csv).

Remaining limitations include stylized vulnerability and road networks, static four-state probabilities, complete day-two observation, a single resource and no estimate of health outcomes. Scenario draws measure policy behavior across hypothetical conditions. Field use would require validated needs, logistics and authority, including evidence that alternative policies improve real service. [Scenario outcomes](results/simulation_cases.csv).
