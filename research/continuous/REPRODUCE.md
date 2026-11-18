# Reproduce the first continuous research batch

Run from the repository root on x86-64 Linux with Python 3.12 and `research/requirements.txt`. The original source snapshots allow offline replay. Source-page retrieval in the examples verifies cached content and source URLs before parsing. [Numeric runtime details](../augmentation/REPRODUCE.md).

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_CORETYPE=Haswell
export NPY_DISABLE_CPU_FEATURES=X86_V4,AVX512_ICL,AVX512_SPR
sha256sum -c research/continuous/coverage/inputs/SHA256SUMS
sha256sum -c research/continuous/earthquake/inputs/SHA256SUMS
mkdir -p work/continuous-replay/{dds,usgs,coverage,ablation,pga-input,pga}
tar -xzf research/continuous/coverage/inputs/openfema_public_responses.tar.gz -C work/continuous-replay/dds
tar -xzf research/continuous/earthquake/inputs/usgs_public_responses.tar.gz -C work/continuous-replay/usgs
python analysis/thesis_historical_coverage.py work/continuous-replay/dds work/continuous-replay/coverage
python analysis/thesis_precoverage_ablation.py work/continuous-replay/ablation
python analysis/thesis_earthquake_auxiliary.py work/continuous-replay/pga-input work/continuous-replay/pga --fetch work/continuous-replay/usgs
jupyter execute research/THESIS_REPRODUCTION.ipynb
```

For the earthquake model alone, pass the retained `research/continuous/earthquake/inputs` folder without the paired fetch argument. The archive also retains the initial narrow query and data to support the recorded coverage diagnostic. The replay never joins the locked FY2025 cohort to obligations.

## Publication and impact evidence replay

```bash
sha256sum -c research/continuous/BATCH2_SHA256SUMS
mkdir -p work/continuous-replay/{filing-snapshot,impact-snapshot,filings,impacts}
tar xf research/continuous/filings/inputs/federal_register_public_responses.tar.gz -C work/continuous-replay/filing-snapshot
tar xf research/continuous/impacts/inputs/noaa_impact_public_responses.tar.gz -C work/continuous-replay/impact-snapshot
python analysis/thesis_declaration_publication.py work/continuous-replay/filing-snapshot work/continuous-replay/filings
python analysis/thesis_earthquake_impact_audit.py work/continuous-replay/impact-snapshot work/continuous-replay/impacts
```
