# AURORA-VISION Assets

This directory contains static assets for the AURORA-VISION dashboard and documentation.

## Directory Structure

```
assets/
├── images/           ← Architecture diagrams, screenshots, banners
├── icons/            ← SVG icons for the Dash UI
├── fonts/            ← Custom fonts for the Vision Ops Center dashboard
├── css/              ← Custom CSS overrides for Dash components
└── demo_videos/      ← Short demo videos for testing (not tracked by git)
```

## Adding Assets

- Place Dash UI static files (CSS, images, fonts) directly in `assets/`
  so Dash serves them automatically at `/assets/<filename>`.
- Sub-directories are only for organization; use relative paths in code.
- Demo videos are excluded from git via `.gitignore` (`assets/demo_videos/`).
