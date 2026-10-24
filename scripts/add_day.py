#!/usr/bin/env python3
"""Create a portfolio lesson shell and optionally attach a self-contained HTML animation.
Run from the ai-infra-learning repository root. No third-party Python dependencies.
"""
import argparse
from pathlib import Path
import shutil
from html import escape

BASE = Path(__file__).resolve().parents[1]
DOCS = BASE / 'docs'
TEMPLATE = BASE / 'templates' / 'day-template.html'
REPO = 'https://github.com/ZachJon1/ai-infra-learning'


def main():
    ap = argparse.ArgumentParser(description='Add a static lesson page without modifying existing lab files')
    ap.add_argument('--day', type=int, required=True, help='Roadmap day, from 2 to 220')
    ap.add_argument('--title', required=True)
    ap.add_argument('--lab', required=True, help='Exact existing GitHub folder, e.g. day2-gpu-benchmarking')
    ap.add_argument('--subtitle', default='Architecture, reproducible implementation, troubleshooting, and engineering notes.')
    ap.add_argument('--animation', type=Path, help='Optional existing standalone HTML animation to embed')
    args = ap.parse_args()
    if not (2 <= args.day <= 220):
        ap.error('Day must be between 2 and 220. Day 1 already has a custom page.')
    if '/' in args.lab or '\\' in args.lab or args.lab in ('.', '..'):
        ap.error('--lab must be one GitHub directory name, not a path')
    dest = DOCS / 'days' / f'day-{args.day:02d}' / 'index.html'
    if dest.exists():
        ap.error(f'{dest} already exists. Edit it manually rather than overwriting it.')
    if args.animation:
        if not args.animation.is_file() or args.animation.suffix.lower() != '.html':
            ap.error('--animation must point to an existing .html file')
        animate_dest = DOCS / 'animations' / f'day-{args.day:02d}.html'
        if animate_dest.exists():
            ap.error(f'{animate_dest} already exists')
        animate_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.animation, animate_dest)
        section = f'''<section class="visual-wrap" aria-label="Interactive animation">
            <div class="visual-header"><div><h2>Interactive visual explainer</h2><p>Explore the animated lesson and its controls.</p></div>
            <a class="btn" href="../../animations/day-{args.day:02d}.html" target="_blank" rel="noopener noreferrer">Open full-screen ↗</a></div>
            <div class="visual-shell"><iframe src="../../animations/day-{args.day:02d}.html" title="{escape(args.title,quote=True)} interactive explainer" sandbox="allow-scripts allow-same-origin" loading="lazy" allowfullscreen></iframe></div>
            <div class="note">This visualization is a conceptual explanation. The GitHub lab and actual results are documented separately below.</div>
            </section>'''
    else:
        section = '<div class="note">No HTML animation has been added for this day yet. The linked hands-on lab can still be explored below.</div>'
    raw = TEMPLATE.read_text(encoding='utf-8')
    replacements = {
        '__DAY__': f'{args.day:02d}',
        '__TITLE__': escape(args.title, quote=True),
        '__SUBTITLE__': escape(args.subtitle, quote=True),
        '__LAB_URL__': f'{REPO}/tree/main/{escape(args.lab,quote=True)}',
        '__ANIMATION_SECTION__': section,
    }
    for key,val in replacements.items():
        raw = raw.replace(key,val)
    dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(raw,encoding='utf-8')
    print(f'Created {dest.relative_to(BASE)}')
    print('Next: replace placeholder notes with the corresponding real lab details.')
    print('Next: add a link on the homepage or roadmap before publishing the new page.')

if __name__ == '__main__':
    main()
