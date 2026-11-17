# Research architecture and evidence boundaries

```mermaid
flowchart TD
  A[Authoritative source records and versions] --> B[Date, geography, units, missingness and provenance audit]
  B --> C[Verified evidence with explicit uncertainty]
  B --> D[Unverified candidates: research only]
  C --> E[Historical funding predictor: frozen Track A]
  E --> F[Accounting prediction with validation and uncertainty labels]
  C --> G[Independently justified need and vulnerability]
  G --> H[Budget and fairness constraints]
  H --> I[Allocation policy comparison]
  F --> J[Human evidence review]
  I --> J
  J --> K[Accountable human decision]
```

This is the selected separation of responsibilities, not evidence that every
component is operationally validated. Funding is an accounting prediction, not
the need variable driving allocation. Track A remains retrospective. The current
allocation implementation uses synthetic needs; real-need linkage is incomplete.
Unverified severity/exposure candidates cannot enter an accepted model. Missing
observations remain missing and can trigger abstention or human review.
