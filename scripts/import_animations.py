#!/usr/bin/env python3
"""Import standalone interactive lesson HTMLs (Days 2–220) into the portfolio.

Works with a folder or ZIP archive. HTML filenames / parent folders should contain
'Day 02', 'day-02', 'day_2', etc. For other names use a CSV manifest with columns
'day,filename' pointing at relative paths inside the archive/source directory.

Preserves Day 1's custom page and all existing laboratory source folders.
Only creates new lesson and animation files; refuses to overwrite existing pages.
Updates roadmap links and published counts after a successful import.
Python standard library only. No remote requests or GitHub writes.
"""
import argparse
import csv
from html import escape
import io
import json
from pathlib import Path, PurePosixPath
import re
import sys
from zipfile import ZipFile, is_zipfile

BASE = Path(__file__).resolve().parents[1]
DOCS = BASE / 'docs'
ROADMAP_JSON = DOCS / 'data' / 'roadmap.json'
ROADMAP_HTML = DOCS / 'roadmap' / 'index.html'
HOME_HTML = DOCS / 'index.html'
TEMPLATE = BASE / 'templates' / 'day-template.html'
REPO = 'https://github.com/ZachJon1/ai-infra-learning'
DAY_PATTERN = re.compile(r'(?i)(?<![a-z0-9])day[\s_.-]*0*(\d{1,3})(?!\d)')


def read_sources(input_path):
    """Return path->HTML bytes without extracting zip members onto disk."""
    sources = {}
    if input_path.is_dir():
        for file in sorted(input_path.rglob('*')):
            if file.is_file() and file.suffix.lower() in ('.html', '.htm'):
                sources[file.relative_to(input_path).as_posix()] = file.read_bytes()
    elif input_path.is_file() and is_zipfile(input_path):
        with ZipFile(input_path) as z:
            for info in z.infolist():
                name = info.filename.replace('\\', '/')
                if info.is_dir() or PurePosixPath(name).suffix.lower() not in ('.html', '.htm'):
                    continue
                if info.file_size > 20_000_000:
                    raise ValueError(f'HTML exceeds 20 MB cap: {name}')
                sources[name] = z.read(info)
    else:
        raise ValueError('Provide a directory or ZIP of interactive .html files')
    return sources


def choose_days(sources, manifest_file=None):
    found = {}
    unidentified = []
    if manifest_file:
        with manifest_file.open(newline='', encoding='utf-8-sig') as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            day = int(row['day'])
            name = row['filename'].strip().replace('\\', '/')
            if name not in sources:
                raise ValueError(f'Manifest file not found in source: {name}')
            if day in found:
                raise ValueError(f'Duplicate mapping for Day {day}')
            found[day] = name
        unidentified = [s for s in sources if s not in found.values()]
    else:
        for name in sorted(sources):
            # Prefer explicit name/folder indicators. Never guess day numbers from the topic.
            # Prefer filename. Archive parent folders may contain ranges such as
            # day1_to_day38_animations; these are not separate day identifiers.
            filename = PurePosixPath(name).name
            matches = [int(n) for n in DAY_PATTERN.findall(filename)]
            if not matches:
                matches = [int(n) for n in DAY_PATTERN.findall(str(PurePosixPath(name).parent))]
            unique = set(matches)
            if len(unique) != 1:
                unidentified.append(name)
                continue
            day = matches[0]
            if day in found:
                raise ValueError(f'Two files match Day {day}: {found[day]} and {name}. Use a manifest.')
            found[day] = name
    return found, unidentified


def new_page(day, meta):
    raw = TEMPLATE.read_text(encoding='utf-8')
    topic = str(meta['topic']).replace('`', '')
    name = escape(topic, quote=True)
    number = f'{day:02d}'
    animation = f'''<section class="visual-wrap" aria-label="Interactive animation">
      <div class="visual-header"><div><h2>Interactive visual explainer</h2><p>Control the animation to explore the system at your own pace.</p></div>
      <a class="btn" href="../../animations/day-{number}.html" target="_blank" rel="noopener noreferrer">Open full-screen ↗</a></div>
      <div class="visual-shell"><iframe src="../../animations/day-{number}.html" title="Day {day}: {name} interactive explainer" sandbox="allow-scripts allow-same-origin" loading="lazy" allowfullscreen></iframe></div>
      <div class="note">This interactive explainer illustrates the concepts. The engineering implementation and any verified measurements are in the original GitHub lab.</div>
    </section>'''
    lab = meta.get('lab_directory')
    lab_url = f'{REPO}/tree/main/{escape(lab,quote=True)}' if lab else REPO
    replacements = {
        '__DAY__': number,
        '__TITLE__': name,
        '__SUBTITLE__': escape(f'Visual explanation for Day {day} of the AI Infrastructure & ML Systems roadmap.', quote=True),
        '__LAB_URL__': lab_url,
        '__ANIMATION_SECTION__': animation,
    }
    for k,v in replacements.items():
        raw = raw.replace(k,v)
    # No invented implementation/results: direct the reader to authentic lab material.
    raw = raw.replace('Write what the lab aims to demonstrate. Match the real implementation, not only the animation.',
                      'The original lab contains the learning objectives and the context for this visual explanation. See the linked GitHub README for the precise objectives.')
    raw = raw.replace('Describe the setup and key commands here. Link code, YAML, tests, and configurations stored in the existing GitHub lab folder.',
                      'Explore the corresponding GitHub lab folder for the actual commands, configurations, code, and execution instructions. This site does not substitute conceptual animations for working code.')
    raw = raw.replace('Add genuine logs, measurements, failure/recovery evidence, and limitations. Do not claim results before running the experiment.',
                      'Check the source lab for any recorded logs, measurements, troubleshooting steps, and known limitations. Only measured or otherwise observed results should be reported as evidence.')
    raw = raw.replace('Capture operational reasoning, trade-offs, and a concise interview takeaway.',
                      'Review the source lab and connect the animated concept to operational decisions, failure modes, and interview scenarios.')
    return raw


def roadmap_link(page, day, meta):
    number = f'{day:03d}'
    topic = escape(str(meta['topic']).replace('`', ''))
    # Existing roadmap contains each published/source lab as one flat <a> card.
    pattern = re.compile(r'<a class="day(?: day-featured)?"[^>]*>\s*<div class="day-top">\s*<span class="day-num">Day\s+' + number + r'</span>.*?</a>', re.S)
    replacement = (f'<a class="day day-featured" href="../days/day-{day:02d}/">'
                   f'<div class="day-top"><span class="day-num">Day {number}</span>'
                   f'<span class="day-kind">Interactive</span></div>'
                   f'<div class="day-title">{topic}</div></a>')
    revised, n = pattern.subn(lambda _: replacement, page, count=1)
    if n != 1:
        raise ValueError(f'Could not locate Day {day} card in roadmap HTML; no changes written')
    return revised


def publish_count(n):
    home = HOME_HTML.read_text(encoding='utf-8')
    home = re.sub(r'<strong>\d{2,3}</strong><small>Interactive lesson[s]? (?:published in this preview|in this preview)</small>',
                  f'<strong>{n:02d}</strong><small>Interactive lessons in this preview</small>', home)
    HOME_HTML.write_text(home, encoding='utf-8')


def update_roadmap_note(html, n):
    note = ('<div class="note"><strong>Progress transparency:</strong> '
            f'{n} interactive lesson page(s) are available in this preview. '
            'Other roadmap entries link to source labs where available or remain planned topics. '
            'Interactive visuals illustrate concepts; the source repository records actual work and evidence.'            '</div>')
    pattern = re.compile(r'<div class="note"><strong>Progress transparency:</strong>.*?</div>', re.S)
    result, count = pattern.subn(note, html, count=1)
    return result if count else html


def main():
    p = argparse.ArgumentParser(description='Bulk import interactive HTML explainers into existing light-theme portfolio')
    p.add_argument('--input', type=Path, required=True, help='Folder or ZIP containing HTML animations')
    p.add_argument('--manifest', type=Path, help='Optional CSV: day,filename (relative name inside ZIP or folder)')
    p.add_argument('--dry-run', action='store_true', help='Show what would be imported, without changing files')
    args = p.parse_args()
    try:
        sources = read_sources(args.input)
        days, unidentified = choose_days(sources, args.manifest)
        if not sources:
            raise ValueError('No HTML files found')
        if unidentified:
            print('Unmatched HTML filenames (need Day N in path or CSV manifest):')
            for name in unidentified: print(f'  - {name}')
        data = json.loads(ROADMAP_JSON.read_text(encoding='utf-8'))
        entries = {int(x['day']):x for x in data}
        html = ROADMAP_HTML.read_text(encoding='utf-8')
        available = []
        skipped = []
        for day,name in sorted(days.items()):
            if day == 1:
                skipped.append((day, 'Custom Day 1 page already exists'))
                continue
            if day not in range(2,221):
                skipped.append((day, 'Outside the 220-day roadmap'))
                continue
            dest_page = DOCS / 'days' / f'day-{day:02d}' / 'index.html'
            dest_animation = DOCS / 'animations' / f'day-{day:02d}.html'
            if dest_page.exists() or dest_animation.exists():
                skipped.append((day, 'Page or animation already exists — no overwrite'))
                continue
            if not re.search(br'(?i)<!doctype\s+html|<html\b', sources[name][:16000]):
                skipped.append((day, 'Not recognized as a standalone HTML page'))
                continue
            html = roadmap_link(html, day, entries[day])
            available.append((day,name,dest_page,dest_animation))
        for day,name,*_ in available:
            print(f'  Day {day:02d} ← {name}')
        for day,reason in skipped:
            print(f'  SKIP Day {day:02d}: {reason}')
        print(f'Importable: {len(available)} / {len(sources)} files')
        if args.dry_run:
            return
        for day,name,page_dest,animation_dest in available:
            page_dest.parent.mkdir(parents=True, exist_ok=True)
            animation_dest.parent.mkdir(parents=True, exist_ok=True)
            animation_dest.write_bytes(sources[name])
            page_dest.write_text(new_page(day, entries[day]), encoding='utf-8')
            entries[day]['interactive_published'] = True
        if available:
            count = sum(x.get('interactive_published', False) for x in data)
            ROADMAP_HTML.write_text(update_roadmap_note(html, count), encoding='utf-8')
            ROADMAP_JSON.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n',encoding='utf-8')
            publish_count(count)
            print(f'Updated roadmap links; {count} interactive lesson(s) available.')
        if unidentified:
            print('Unmatched files were not imported. Provide --manifest CSV for explicit mapping.', file=sys.stderr)
    except (ValueError, KeyError, OSError) as e:
        print(f'ERROR: {e}', file=sys.stderr)
        sys.exit(2)

if __name__ == '__main__':
    main()
