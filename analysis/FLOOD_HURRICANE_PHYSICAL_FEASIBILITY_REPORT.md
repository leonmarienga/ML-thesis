# Physical flood and hurricane enrichment — initial USGS gauge feasibility

Research run: https://github.com/leonmarienga/ML-thesis/actions/runs/41401467244

## Outcome-blind county-linked USGS pilot
Of 971 FEMA master disasters, **92** are labeled Flood in fiscal 2019–2024. Existing NOAA event-to-county audit provides counties for **87** of those. The pilot deterministically selected **12** FEMA Flood records, using the first county FIPS for each, irrespective of funding. USGS NWIS site / daily discharge observations were retrieved for **6/12** records. Case-level station IDs and discharge maxima are saved in the run artifact.

A matched gauge **is not automatically attributable to the FEMA flood**. A gauge may be upstream/downstream of the damaged area or unrelated to the relevant watershed. Discharge needs reference distributions or return periods, rather than raw peak cfs alone.

## Hurricane foundation
Earlier outcome-blind NHC advisory extraction recovered storm issue-time measurements for **113/123 tropical records** (incident-year source matching). Only **41** have a recent NHC advisory within 72h at conservative declaration-date T0; archived header times were validated on a 25-bulletin sample. These existing meteorological features are not yet connected to localized exposure or precipitation.

## What is and is not completed
- USGS gauge API retrieval and initial county association: **feasible in sample**, uncertain hydrological match.
- NASA IMERG time-bounded rainfall extraction: **not completed** (requires validated geographic footprints, data acquisition/provenance and possibly Earthdata access).
- Flood inundation extent from radar/satellite: **not completed**.
- Exposed housing and infrastructure specifically inside flood/hurricane footprints: **not completed**.
- Independent economic losses and source timing: **not completed**.
- Model improvement vs accepted router/amount stack: **not tested in this pilot**.

## Next acceptance gates
1. Validate affected county/region and river basin/station location, avoid arbitrary first station/county.
2. Calculate same-gauge pre-event historical discharge percentiles, normalized peak-to-baseline, and observation completeness.
3. Attach independently observed precipitation (NASA IMERG or verified NOAA rain-gauge data) at the matched spatio-temporal footprint.
4. Reuse archived NHC *issue-time* wind/pressure, add geographically localized hurricane exposure and incident-time rainfall.
5. Create target-blind, one-row-per-FEMA-disaster feature records and raw station/footprint evidence tables.
6. Report missingness by year/hazard and verify as-of cutoffs separately for retrospective and T0/T+72 research.
7. Compare frozen baseline vs severity-only vs exposure-only vs combined with strict same-year test cases; report rare band results and uncertainty. Do not modify accepted 830/912 router or ~$5.015M MAE model.
