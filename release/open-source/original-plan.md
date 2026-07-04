# Original Open Source Release Plan

This file preserves the original plan from the release-planning chat. The implementation feature list is maintained in:

- `PUBLIC_RELEASE_PLAN.md`
- `release/open-source/features.md`

## Original Plan

You should treat this as a clean public-source release, not simply "make the current private repo public".

The repository already contains public-release scaffolding:

- `LICENSE`
- `NOTICE`
- `CITATION.cff`
- `README.md`
- `CONTRIBUTING.md`
- `SECURITY.md`
- `PUBLIC_RELEASE_CHECKLIST.md`

The current licence is MIT and asks users to acknowledge Mahan Yoldashkhan and `1Dtransport.com`.

## Recommended Approach

1. Keep the private repo as the working/internal repo.
2. Create a new clean public GitHub repository. Preferred names:
   - `boef`
   - `boef-rail`
   - `boef-engineering`
   - `beam-on-elastic-foundation`
3. Export only the public-safe source tree into the new repository.
4. Do not copy old Git history, because old history may contain paper/reference files and local data paths.
5. Run tests before release so functionality is not lost.
6. Push the clean repository publicly only after secret and private-data scans.

## Licence Direction

Use MIT License plus `NOTICE` and `CITATION.cff`.

MIT allows use, copying, modification, distribution, and commercial use. It legally requires preservation of the copyright and licence notice.

The attribution language referring to Mahan Yoldashkhan and `1Dtransport.com` is an attribution request, not an extra legal restriction on MIT.

If strict attribution is required as a legal condition, seek legal advice before release because that may stop the licence being standard open source.

## Main Release Steps

1. Freeze the private repo temporarily.
2. Clean and verify the current working tree.
3. Decide what is public and what is excluded.
4. Run secret and private-data scans.
5. Run BOEF functionality checks.
6. Build a clean public export.
7. Create the public GitHub repository.
8. Push the clean export.
9. Configure GitHub settings.
10. Add the first release tag.
11. Update `1Dtransport.com`.
12. Keep private and public repositories separated.

## Latest Scope Decision

Keep `AGENTS.md` public. Merge any public-safe recommendations from `python.md` into `AGENTS.md`, but do not publish `python.md`.
