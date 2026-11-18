# Auxiliary physical simulation and earthquake transfer

The study evaluated an independently measured physical endpoint. It matched published earthquake source and site properties to instrument-recorded peak ground acceleration, then tested whether physics-derived synthetic training improved later-episode predictions. The test results support retaining the trial as auxiliary evidence. They do not establish a transfer gain large enough for the proposed integration gate.

## Source and support audit

The AtlasCat bulk endpoint returned HTTP 403. The official ComCat and ShakeMap station archives supported a reproducible alternative. The 2005–24 query provided ten qualified episodes after correcting location-prefixed channel parsing. Five episodes remained for forward evaluation after the earlier-training requirement. Extending the same physical and quality screen back to January 1998, after the publication of BJF97, yielded 608 admissible instrument measurements from 13 earthquakes. Twelve earthquakes met the five-station support threshold. Six screened moment-magnitude representatives were below it. The full query and station payloads are byte-identified and retained. [Protocol](../EARTHQUAKE_PILOT_PROTOCOL.md), [initial support diagnostic](narrow_screen/pga_fold_support.csv), [source manifest](inputs/earthquake_source_manifest.csv), [episode dispositions](inputs/earthquake_sources.csv).

The final test used seven independent later events and 143 held measurements. Each measured PGA value combines two valid horizontal channels at the same station instrument location. Instruments with nonzero quality flags and macroseismic reports were excluded. The source provides modeled or measured Vs30 and finite-fault distance from a retrospective reconstruction. Some screened events are offshore or tectonically ambiguous. The effect of excluding poorer-quality observations and the physical transportability of BJF97 are unresolved. Treat station-level intervals as descriptive because readings within an event share a source, event terms, and spatially correlated ground motion. [Station inputs](inputs/instrumental_pga.csv), [fold support](results/pga_fold_support.csv).

## Fixed physical and synthetic comparisons

| Policy | Equal-episode MAE in log PGA | Pooled error at measured PGA at least 0.10g | Reference stress-band coverage |
| --- | ---: | ---: | ---: |
| Real-only Ridge | .5383 | .6319 | 75.5% |
| BJF97 physics only | .6992 | .4058 | 69.2% |
| BJF97 plus earlier real intercept | .5619 | .6564 | 79.7% |
| Hybrid with 100 simulations | .5216 | .6093 | 78.3% |
| Hybrid with 500 simulations | .5216 | .6103 | 78.3% |
| Hybrid with 1,000 simulations | .5223 | .6111 | 77.6% |
| Hybrid with 5,000 simulations | .5222 | .6110 | 77.6% |

The 100-scenario hybrid reduced equal-episode error by **3.11%**. The paired 95% episode-bootstrap interval for its absolute change was **−.0518 to +.0108**. All synthetic sizes had similar error and intervals spanning zero. The prior's total fitting weight stayed equivalent to one real training episode. Increasing volume primarily changed how finely the same published physical relationship was sampled. Its additional rows supplied no independent observations. The physics-only arm was strongest on the 38 measured high-PGA records and had the largest equal-episode overall error. The published sigma stress bands covered roughly 69–80% of measured records at the nominal 90% reference multiplier. These bands carry no calibrated local coverage claim. [Metrics](results/pga_metrics.csv), [paired episode bootstrap](results/pga_paired_bootstrap.csv), [simulated prior scenarios](results/pga_scenarios.csv.gz).

The experiment assigns no synthetic FEMA obligations, needs, or damages. A defensible next auxiliary test would pair separately observed building or infrastructure impacts with authenticated physical exposures and address mechanism and site domain before attempting cross-task transfer. The current matched funding dataset still has the same number of real catastrophic episodes. [USGS archive](https://www.usgs.gov/data/shakemap-atlas-v4-and-atlascat), [BJF97](https://pubs.usgs.gov/publication/70020212).
