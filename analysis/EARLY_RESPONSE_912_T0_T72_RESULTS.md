# Research Track B: 912-case T0 vs T+72h early-response chronological experiment

Executed successfully: https://github.com/leonmarienga/ML-thesis/actions/runs/41393614710

## Population and design
The original 971 FEMA records include **912 non-Biological disasters**. The trained models use all preceding fiscal years; only fiscal **2019–2024** is scored, because those years form the pre-defined later chronological evaluation period. The matched evaluated sample has **376 disasters**: 266 below $100K; 50 at $100K–$1M; 47 at $1M–$50M; 9 at $50M–$200M; 1 at $200M–$500M; and 3 above $500M.

Feature configurations are: (1) declaration metadata only, (2) + T0 action-time filtered FEMA missions, (3) + recent NHC advisory wind/pressure at T0, (4) + T+72 mission proxies, (5) + recent NHC at T+72. ExtraTrees regressor (log-dollar, 250 trees, min leaf 3) and separate six-class ExtraTrees classifier (250 trees, min leaf 2), fixed architecture, no target features in source enrichment.

## All-band results

| Variant | MAE dollars | RMSE dollars | R² | Band accuracy | Balanced accuracy | Macro F1 |
|---|---:|---:|---:|---:|---:|---:|
| Metadata | $19,921,808 | $190,727,570 | -0.01085 | 47.61% | 26.50% | 0.2345 |
| Metadata + T0 missions | $20,114,641 | $190,694,707 | -0.01050 | 51.86% | 27.46% | 0.2454 |
| Metadata + T0 missions + recent NHC | $19,695,723 | $189,869,243 | -0.00177 | 54.79% | 27.69% | 0.2611 |
| Metadata + T+72 missions | $19,734,715 | $188,416,707 | +0.01350 | 62.50% | 31.54% | 0.2952 |
| Metadata + T+72 missions + recent NHC | **$19,487,524** | **$188,147,885** | **+0.01631** | **62.77%** | **33.95%** | **0.3168** |

**T+72 with NHC minus metadata MAE:** -$434,284 with exploratory paired bootstrap 95% interval from -$1.591M to +$0.703M; interval crosses zero. T+72 vs T0 both including recent NHC: -$208,198, interval -$1.208M to +$0.885M.

**Rare high bands remain unreliable:** For T+72 including NHC, recall is 3/9 for $50M–$200M, 0/1 for $200M–$500M, and 0/3 for $500M+. Macro recall 33.95% is far below the accepted router's standard. Do not replace it.

## Available source data
Across the 912 non-Biological records, action-time filtered t0 mission assignments cover 36 disasters; by T+72 cover 212. A ≤72h recent NHC advisory occurs for 41 at T0 and 27 at T+72. Note the last two are different cutoff-time freshness checks, not cumulative counts.

## Essential limitations
1. Exact original FEMA declaration issuance times and versioned values are not authenticated. The code uses midnight UTC of the declaration date. The initial metadata fields could have been updated after that date.
2. All present-day MissionAssignments rows have `lastRefresh` after t0, so filtering by `dateReceived` is not a fully reconstructed prior record version.
3. Storm identity is matched using the current FEMA title; original declaration title availability remains unverified.
4. Final total obligated funding is a retrospective target, **not** a contemporaneous humanitarian resource-requirement measurement.
5. Chronological 2019–2024 tests were drawn from a historically repeatedly analyzed dataset; these are exploratory, not untouched external validation.
6. Repeated hurricanes across states, extremely few high-value cases, and heavy-tailed dollars make uncertainty high. Paired bootstrap is exploratory and does not cluster by shared storm.
7. The experiment does not evaluate the accepted specialist architecture and must not be compared to the accepted final-model 91.01% accuracy or $5.015M MAE as an equivalent predictor, because data availability and test population differ.

## Research conclusion
The 72-hour version benefits from substantially more observed mission actions than the T0 version, improving average routing performance on the 376 held-out later-fiscal-year records; however, **rare high bands fail the accepted recall requirements and dollar accuracy remains weak**. Preserve the accepted retrospective Track A exactly and develop Track B independently. The next decisive step is to retrieve original dated FEMA records/version snapshots and evaluate new future events separately, while advancing external hazard observations with accurate issue times.
