"""Offline training/evaluation tool — pure analysis logic separated from the realtime pipeline.

The `analysis` package contains the analyzers carved out of `app.vision`, plus
the new offline-only modules: ingest, sidecar store, and the sequential
multi-pass orchestrator. The carried-forward analyzer modules are byte-identical
independent copies of their `app.vision` originals — `app.vision` itself is
left untouched, and the two code paths evolve independently.

See the architecture report for the module boundaries and data model.
"""
