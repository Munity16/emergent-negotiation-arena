# Project Status

Last updated: 2026-07-11

## Deployment

| Item | Status |
|---|---|
| Render Web Service (free instance, Python 3.12.8) | ✅ Completed |
| GitHub auto-deployment from `main` | ✅ Enabled |
| Start command `python ui/app.py` | ✅ Working |
| Public page load (branding + encoding correct) | ✅ Verified |
| Replay backend on deployed service | ✅ Verified |
| Heuristic backend (background simulation thread) on deployed service | ✅ Verified |
| Access from a separate mobile device | ✅ Verified |
| Mobile responsiveness fix (`ui/app.py` CSS, commit `1a93c6c`) | ✅ Pushed — final on-device verification pending |
| Fireworks backend on deployed service | ⏳ Configured for secure runtime use via Render environment settings; final live deployment test not yet run |

Notes:

- Free Render instances sleep after inactivity — expect a cold-start delay
  on the first request after a quiet period.
- The public URL is intentionally not yet published in the documentation; it
  will be added after final mobile testing.
- 84/84 automated tests passing locally.

## Next tasks

1. Final hackathon submission
2. Presentation
3. Demo video

(Also pending: add the verified Render URL to README/SUBMISSION after final
mobile testing, and run the Fireworks live deployment test once the key is
added in Render.)
