# Session handoff

## 2026-08-10

- Review phase fixed the person statistic so the Swedish header keeps `unique_count` as its headline while showing the API's projected `count` with P3's immutable uncertainty band.
- Focused API regression was attempted with `pytest tests/test_review_api_phase5.py -k persons -v`, but collection is blocked because the available Python environment lacks `cv2`.
- Documentation pass aligned the Swedish guide's person-list wording with the corrected projection and found no other stale owner documentation.
