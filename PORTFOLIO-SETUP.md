# AI Infrastructure Lab — 38 Interactive Lessons (Light Theme)

This is a **local-review, ready-to-publish static site**, prepared for the existing repository `https://github.com/ZachJon1/ai-infra-learning`. No remote repository changes or GitHub Pages deployment were made.

## Included

- `docs/index.html`: portfolio homepage, light color palette, featured labs, and interactive-learning categories.
- `docs/animations/index.html`: searchable gallery for all 38 interactive lessons grouped by roadmap phase.
- `docs/animations/day-02.html` … `day-38.html`, plus `linux-to-gpu.html` for Day 1: **unchanged HTML originals** from the ZIP provided by the user.
- `docs/days/day-01/` … `day-38/`: dedicated lesson pages with embedded animation, source GitHub links, and previous/next navigation.
- `docs/roadmap/index.html`: 220-day roadmap with interactive Days 1–38 clearly marked.
- `docs/projects/index.html`: curated engineering capstones and links back to existing source code.
- `scripts/import_animations.py`, `scripts/add_day.py`, `templates/day-template.html`: tools/templates for subsequent lessons.
- `.github/workflows/deploy-pages.yml`: optional static Pages deployment workflow.

**Limitations:** The original Day 1 source folder currently only has a README. Some other day pages currently summarize the roadmap topic and direct visitors to original GitHub lab folders; they do not invent missing benchmarks or test results. Continue enriching these from the original READMEs as desired.

### External animation assets

The original animations were not rewritten. Most animations from Day 3 onward fetch Google Fonts. The **Day 35 animation also fetches GSAP + MotionPathPlugin JavaScript from cdnjs**, so it needs access to those services to work. Avoid opening external animation files you do not trust; the published lesson pages use script-only sandboxed iframes. If offline support is needed, review external dependencies and licenses before bundling them locally.

## Test locally

```bash
cd /path/to/ai-infra-portfolio-38-days
python3 -m http.server 8000 --directory docs
```

Visit `http://localhost:8000/`, open **Interactive lessons**, choose any day, and use the controls in that HTML animation. Test Day 35 with an internet connection. All site links are relative so the same files can be hosted from the repository's project Pages subpath.

## Integrate with your existing GitHub repository

From your local clone of `ai-infra-learning` (do not replace the repository or delete current labs):

```bash
cp -a /path/to/ai-infra-portfolio-38-days/docs ./
cp -a /path/to/ai-infra-portfolio-38-days/scripts ./
cp -a /path/to/ai-infra-portfolio-38-days/templates ./
cp /path/to/ai-infra-portfolio-38-days/PORTFOLIO-SETUP.md ./
git status
```

The website additions are designed to live alongside existing `day*-*` lab folders. Review staged changes and scan all HTML/scripts before publication. Do **not** commit files containing secrets, access keys, private logs, or proprietary datasets.

### Deploy on GitHub Pages

In **Settings → Pages**, choose **Deploy from a branch**, branch `main`, folder `/docs`, and save after committing the website files. Do not also enable the bundled Actions workflow with branch deployment. Alternatively, use **GitHub Actions** publishing and the `.github/workflows/deploy-pages.yml` file, and configure Pages to use GitHub Actions.

The expected eventual URL is `https://zachjon1.github.io/ai-infra-learning/` **only after you explicitly publish**.

## Publish videos separately

Videos have not been uploaded to YouTube. The ZIP contains HTML originals and website files, **not MP4 files**. Once public video URLs are available, link them in the relevant daily lessons.

## Add Day 39 onward

Use `python3 scripts/add_day.py --day 39 --title "GPU-backed inference" --lab <actual-lab-folder> --animation /path/to/day39.html`, or import a new ZIP using `python3 scripts/import_animations.py --input /path/to/new-animations.zip --dry-run` followed by the same command without `--dry-run`. Inspect results before pushing to GitHub. Update the homepage/gallery links when you expand past Day 38.

## Publish-check list

- Verify mobile and desktop interaction for each original animation.
- Verify GitHub lab links, and distinguish educational animations from real empirical results.
- Check your HTML files for external scripts and resource requests, especially Day 35.
- Add video URLs only after actual YouTube publication.
- Update your repository README's Day 31 progress line when publishing, if still outdated.
- Create a GitHub Pages deployment only after reviewing all content.
