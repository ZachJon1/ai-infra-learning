#!/usr/bin/env python3
"""Build Pages indexes from day-N repository labs and optional animations.

Convention: add a lab in a root directory named dayN-topic. Add its optional
standalone animation at docs/animations/day-N.html whenever it is ready. The
Pages workflow runs this script before uploading docs, so generated pages need
not be committed.
"""
from __future__ import annotations

from html import escape
from html.parser import HTMLParser
import json
from pathlib import Path
import re
import sys

BASE = Path(__file__).resolve().parents[1]
DOCS = BASE / 'docs'
ANIMATIONS = DOCS / 'animations'
ROADMAP_JSON = DOCS / 'data' / 'roadmap.json'
ROADMAP_HTML = DOCS / 'roadmap' / 'index.html'
ANIMATIONS_HTML = ANIMATIONS / 'index.html'
HOME_HTML = DOCS / 'index.html'
REPO = 'https://github.com/ZachJon1/ai-infra-learning'
DAY_FILE = re.compile(r'^day-(\d{1,3})\.html$', re.I)

sys.path.insert(0, str(BASE / 'scripts'))
from import_animations import new_page  # noqa: E402


class TitleParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_title = False
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() == 'title':
            self.in_title = True

    def handle_endtag(self, tag):
        if tag.lower() == 'title':
            self.in_title = False

    def handle_data(self, data):
        if self.in_title:
            self.parts.append(data)


def animation_title(path, fallback):
    parser = TitleParser()
    parser.feed(path.read_text(encoding='utf-8', errors='replace')[:50000])
    title = ' '.join(' '.join(parser.parts).split())
    title = re.sub(r'\s*[|–—-]\s*Day\s*\d+.*$', '', title, flags=re.I).strip()
    return title or fallback


def discover_labs(day, preferred=None):
    matches = sorted(p.name for p in BASE.iterdir()
                     if p.is_dir() and re.match(rf'^day0*{day}-', p.name, re.I))
    # The roadmap is the authoritative mapping when a day has multiple folders
    # with the same prefix (for example Day 5's networking lab and web demo).
    if preferred in matches:
        return [preferred]
    return matches


def roadmap_card(page, day, meta, interactive):
    number = f'{day:03d}'
    topic = escape(str(meta['topic']).replace('`', ''))
    if f'href="../days/day-{day:02d}/"' in page:
        return page
    pattern = re.compile(
        rf'<div class="day" aria-label="Day\s+0*{day} curriculum topic">.*?</div></div>', re.S)
    label = 'Interactive' if interactive else 'Lab available'
    replacement = (f'<a class="day day-featured" href="../days/day-{day:02d}/">'
                   f'<div class="day-top"><span class="day-num">Day {number}</span>'
                   f'<span class="day-kind">{label}</span></div>'
                   f'<div class="day-title">{topic}</div></a>')
    revised, count = pattern.subn(lambda _: replacement, page, count=1)
    if count != 1:
        raise ValueError(f'Could not find Day {number} in roadmap HTML')
    return revised


def animation_card(day, title, meta):
    number = f'{day:03d}'
    topic = escape(str(meta['topic']).replace('`', ''))
    return (f'<a class="day day-featured" href="../days/day-{day:02d}/">'
            f'<div class="day-top"><span class="day-num">Day {number}</span>'
            '<span class="day-kind">Open interactive ↗</span></div>'
            f'<div class="day-title">{escape(title)}</div>'
            f'<div class="day-description">{topic}</div></a>')


def add_animation_card(page, day, title, meta):
    card = animation_card(day, title, meta)
    if f'href="../days/day-{day:02d}/"' in page:
        return page
    phase = int(meta['phase'])
    section_pattern = re.compile(
        rf'(<section class="phase" data-phase="{phase}"[^>]*>.*?)(</div></section>)', re.S)
    match = section_pattern.search(page)
    if match:
        section = match.group(0)
        days = [int(row['day']) for row in ROADMAP_ROWS if int(row['phase']) == phase and int(row['day']) in PUBLISHED_DAYS]
        if days:
            section = re.sub(r'<small>Days\s+\d+[–-]\d+</small>',
                             f'<small>Days {min(days):02d}–{max(days):02d}</small>', section, count=1)
        return page[:match.start()] + section[:-len('</div></section>')] + card + '</div></section>' + page[match.end():]

    phase_days = [int(row['day']) for row in ROADMAP_ROWS if int(row['phase']) == phase]
    low, high = min(phase_days), max(phase_days)
    title_text = escape(str(meta['phase_title']))
    section = (f'<section class="phase" data-phase="{phase}" id="phase-{phase}">'
               f'<h2>{phase:02d}. {title_text} <small>Days {low:02d}–{high:02d}</small></h2>'
               f'<div class="days-grid">{card}</div></section>')
    anchor = '</div></main>'
    if anchor not in page:
        raise ValueError('Could not find main section in interactive lessons page')
    return page.replace(anchor, section + anchor, 1)


def replace_lesson_counts(page, count):
    page = re.sub(r'\b\d+(?= interactive lessons?\b)', str(count), page, flags=re.I)
    page = re.sub(r'\b\d+(?= interactive HTML explainers?\b)', str(count), page, flags=re.I)
    page = re.sub(r'\b\d+(?= interactive animations?\b)', str(count), page, flags=re.I)
    page = re.sub(r'\b\d+(?= interactive engineering explainers?\b)', str(count), page, flags=re.I)
    page = re.sub(r'(Interactive explainers · Days 0*\d+[–-])0*\d+',
                  lambda m: m.group(1) + f'{max(1, max(PUBLISHED_DAYS)):02d}', page)
    return page


def update_home_stats(page, lab_count, animation_count):
    page = re.sub(
        r'<strong data-count="labs">\d+</strong>',
        f'<strong data-count="labs">{lab_count}</strong>', page)
    page = re.sub(
        r'<strong data-count="interactive">\d+</strong>',
        f'<strong data-count="interactive">{animation_count}</strong>', page)
    return page


def update_progress_note(page, lab_count, animation_count):
    note = ('<div class="note"><strong>Progress transparency:</strong> '
            f'{lab_count} day lab folder(s) are available; {animation_count} include an interactive animation. '
            'Other roadmap entries remain planned topics. The source repository records the hands-on work and evidence.'
            '</div>')
    pattern = re.compile(r'<div class="note"><strong>Progress transparency:</strong>.*?</div>', re.S)
    revised, found = pattern.subn(note, page, count=1)
    if not found:
        raise ValueError('Could not locate the roadmap progress note')
    return revised


def update_home_phase_cards(page, rows):
    counts = {}
    days = {}
    for row in rows:
        if row['day'] in PUBLISHED_DAYS:
            phase = int(row['phase'])
            counts[phase] = counts.get(phase, 0) + 1
            days.setdefault(phase, []).append(int(row['day']))

    def update(match):
        card = match.group(0)
        found = re.search(r'href="animations/#phase-(\d+)"', card)
        if not found:
            return card
        phase = int(found.group(1))
        if phase not in counts:
            return card
        lo, hi = min(days[phase]), max(days[phase])
        card = re.sub(r'(<span class="card-label">)Days\s+\d+[–-]\d+(</span>)',
                      rf'\g<1>Days {lo:02d}–{hi:02d}\2', card)
        card = re.sub(r'Explore \d+ explainers', f'Explore {counts[phase]} explainers', card)
        return card

    return re.sub(r'<article class="card">.*?</article>', update, page, flags=re.S)


def main():
    global ROADMAP_ROWS, PUBLISHED_DAYS
    rows = json.loads(ROADMAP_JSON.read_text(encoding='utf-8'))
    by_day = {int(row['day']): row for row in rows}
    animations = {}
    for path in sorted(ANIMATIONS.iterdir()):
        match = DAY_FILE.match(path.name)
        if path.is_file() and match:
            day = int(match.group(1))
            if day in animations:
                raise ValueError(f'Multiple animations found for Day {day}')
            animations[day] = path
    # Day 1 predates the day-N filename convention and keeps its original name.
    day_one_animation = ANIMATIONS / 'linux-to-gpu.html'
    if day_one_animation.is_file() and 1 not in animations:
        animations[1] = day_one_animation

    if any(day not in by_day for day in animations):
        invalid = sorted(day for day in animations if day not in by_day)
        raise ValueError(f'Animation day(s) outside the roadmap: {invalid}')

    ROADMAP_ROWS = rows
    PUBLISHED_DAYS = set(animations)
    lab_directories = {}
    for day, meta in by_day.items():
        labs = discover_labs(day, meta.get('lab_directory'))
        if len(labs) > 1:
            raise ValueError(f'Multiple lab folders match Day {day}: {labs}')
        if labs:
            lab_directories[day] = labs[0]
            meta['lab_directory'] = labs[0]
    PUBLISHED_LAB_DAYS = set(lab_directories)
    roadmap = ROADMAP_HTML.read_text(encoding='utf-8')
    catalog = ANIMATIONS_HTML.read_text(encoding='utf-8')

    for day in sorted(PUBLISHED_LAB_DAYS | PUBLISHED_DAYS):
        meta = by_day[day]
        page_path = DOCS / 'days' / f'day-{day:02d}' / 'index.html'
        animation_path = animations.get(day)
        title = animation_title(animation_path, meta['topic']) if animation_path else meta['topic']
        if not page_path.exists():
            lab = meta.get('lab_directory')
            lab_url = f'{REPO}/tree/main/{lab}' if lab else REPO
            page_path.parent.mkdir(parents=True, exist_ok=True)
            page_path.write_text(new_page(
                day, {**meta, 'topic': title, 'lab_directory': lab},
                animation_available=animation_path is not None), encoding='utf-8')
        roadmap = roadmap_card(roadmap, day, meta, interactive=animation_path is not None)
        if animation_path is not None:
            catalog = add_animation_card(catalog, day, title, meta)
        meta['interactive_published'] = animation_path is not None

    animation_count = len(PUBLISHED_DAYS)
    lab_count = len(PUBLISHED_LAB_DAYS)
    home = HOME_HTML.read_text(encoding='utf-8')
    home = update_home_stats(home, lab_count, animation_count)
    home = update_home_phase_cards(replace_lesson_counts(home, animation_count), rows)
    HOME_HTML.write_text(home, encoding='utf-8')
    roadmap = update_progress_note(replace_lesson_counts(roadmap, animation_count), lab_count, animation_count)
    ROADMAP_HTML.write_text(roadmap, encoding='utf-8')
    ANIMATIONS_HTML.write_text(replace_lesson_counts(catalog, animation_count), encoding='utf-8')
    ROADMAP_JSON.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'Synchronized {lab_count} lab pages and {animation_count} interactive lessons.')


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError) as error:
        print(f'ERROR: {error}', file=sys.stderr)
        sys.exit(2)
