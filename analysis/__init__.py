"""Offline training/evaluation tool — pure analysis logic separated from the realtime pipeline.

The `analysis` package contains the analyzers carved out of `app.vision`, plus
the new offline-only modules: ingest, sidecar store, and the sequential
multi-pass orchestrator. The realtime PoC uses thin shim modules in `app.vision`
that re-export from here, so both code paths share the same logic.

See the architecture report for the module boundaries and data model.
"""
