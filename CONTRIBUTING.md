# Contributing to USV-AUV-delay

Thanks for helping improve this repository.

## Setup

```bash
git clone https://github.com/Hugh41/USV-AUV-delay.git
cd USV-AUV-delay
pip install -r requirements.txt

# Optional but recommended if you plan to run DSAC-T experiments
cd DSAC-v2
pip install -e .
cd ..
```

## Recommended workflow

1. Create a focused branch from `main`.
2. Keep each pull request scoped to one topic when possible.
3. For code changes, rerun the smallest relevant script or experiment that exercises your update.
4. For figure or README changes, include a short note or screenshot in the pull request when it helps reviewers.

Example:

```bash
git switch -c your-branch-name
# edit files
git add README.md CONTRIBUTING.md
git commit -m "docs: improve contribution guide"
git push -u origin your-branch-name
```

## Pull request checklist

- Explain what changed and why.
- List the command(s) used to validate the change, if any.
- Mention any output directories created by the update, such as `delay_comparison_results/` or model checkpoints.
- Avoid committing large generated artifacts unless they are intentionally part of the contribution.

## Good first contributions

- Clarify README instructions or usage examples.
- Improve figure-generation scripts and plot labels.
- Fix small bugs in training, evaluation, or visualisation scripts.
- Add notes that make experiment reproduction easier.
