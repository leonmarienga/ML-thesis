# Baseline clean-environment replay, 2026-11-28

Baseline commit: b0b6edb229d112dd6d11896a512bf1112414e4f6. Isolated CPython 3.12.14 Linux x86-64 virtual environment, with all 76 observed CI dependency versions installed under constraints and `pip check` clean. The baseline GitHub Actions reference uses 3.12.16, so this is a separate successful environment, not an identical runner image.

- All 24 pinned input hashes passed.
- Phase 50 rebuilt three byte-identical output CSVs, including 67 exactly matched windows.
- All 50 unmodified code cells ran sequentially and outputs were captured in a new notebook. SHA-256: `253f5c8c5d97b2c3b5d42b12798b0d94cc767a2c59cf7c00a9d5bbe84df53090`.
- The normal Jupyter subprocess failed at socket startup in this environment (both TCP and IPC). A socket-free in-process IPython runner executed the original cells with no changes and passed notebook validation. It does not replace a standard Jupyter run as a final reviewer requirement and does not establish byte identity of notebook renderings.
- The original standalone CLI ran on DR-4722 and synthetic scenario 1 in the clean environment, returned JSON and passed presence checks for the record, six probabilities and a distinct simulation object.
- Earlier baseline GitHub Actions ran the standard notebook and the continuous Batches 1–14. This local run is not a fresh execution of those continuous workflows or of a new integrated notebook. Actual browser and keyboard QA of the offline interface, and the final expanded release-wide clean pass, remain outstanding.

The FY2025 outcomes were neither opened nor evaluated. The retained target-free input archive was only checksum verified with the other inputs.
