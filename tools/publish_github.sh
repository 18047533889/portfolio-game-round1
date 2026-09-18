#!/usr/bin/env bash
# Run on YOUR computer after `gh auth login`; never paste a token into chat.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
OWNER="18047533889"
NAME="${1:-portfolio-game-round1}"
if [[ ! "$NAME" =~ ^[A-Za-z0-9._-]+$ ]]; then
  printf '%s\n' 'Invalid repository name.' >&2; exit 1
fi
FULL="$OWNER/$NAME"
command -v gh >/dev/null || { printf '%s\n' 'GitHub CLI is missing. Install gh, then run: gh auth login' >&2; exit 2; }
command -v git >/dev/null || { printf '%s\n' 'Git is missing.' >&2; exit 2; }
gh auth status --hostname github.com >/dev/null 2>&1 || {
  printf '%s\n' 'Authenticate locally first: gh auth login --hostname github.com' >&2; exit 2;
}
LOGIN="$(gh api user --jq .login)"
if [[ "$LOGIN" != "$OWNER" ]]; then
  printf 'Refusing: authenticated as %s, expected %s.\n' "$LOGIN" "$OWNER" >&2; exit 1
fi
# Creates only a repository in this local project folder; never modifies other repos.
if [[ ! -d .git ]]; then git init -b main; fi
if [[ "$(git rev-parse --show-toplevel)" != "$ROOT" ]]; then
  printf '%s\n' 'Refusing to use a parent repository.' >&2; exit 1
fi
if ! git config user.name >/dev/null; then git config --local user.name "$LOGIN"; fi
if ! git config user.email >/dev/null; then
  ID="$(gh api user --jq .id)"
  git config --local user.email "$ID+$LOGIN@users.noreply.github.com"
fi
# Credential helper stores no token here; Git asks the already-authorized gh CLI.
git config --local credential.https://github.com.helper '!gh auth git-credential'
if git remote get-url origin >/dev/null 2>&1; then
  ORIGIN="$(git remote get-url origin)"
  if [[ "$ORIGIN" != "https://github.com/$FULL.git" && "$ORIGIN" != "git@github.com:$FULL.git" && "$ORIGIN" != "https://github.com/$FULL" ]]; then
    printf '%s\n' 'Refusing to overwrite an unrelated origin.' >&2; exit 1
  fi
fi
# Fixed allowlist: excludes datasets, .env, private keys, generated caches and zips.
git add -- .gitignore .github README.md requirements.txt requirements-core.txt requirements-dev.txt pytest.ini \
  configs docs reports research src submission teacher_reference tests tools
if git diff --cached --name-only | grep -E '(^|/)(\.env([.]|$)|id_rsa|id_ed25519)|\.(pem|key|p12|pfx)$' >/dev/null; then
  printf '%s\n' 'Sensitive-looking staged path detected. Nothing pushed.' >&2; exit 1
fi
if ! git diff --cached --quiet; then git commit -m 'feat: standalone Round 1 submission and research toolkit'; fi
if gh repo view "$FULL" --json isPrivate --jq .isPrivate > /dev/null 2>&1; then
  PRIVATE="$(gh repo view "$FULL" --json isPrivate --jq .isPrivate)"
  [[ "$PRIVATE" == 'true' ]] || { printf '%s\n' 'Refusing to publish coursework to an existing PUBLIC repository.' >&2; exit 1; }
  git remote get-url origin >/dev/null 2>&1 || {
    printf '%s\n' 'A repository with this name already exists. Choose another name; it was not modified.' >&2; exit 1;
  }
  # A normal fast-forward push only. No force, deletion, or branch-protection change.
  git push -u origin HEAD:main
else
  gh repo create "$FULL" --private --description 'MAFS5310 Round 1: standalone submission, research and robustness tests' \
    --source . --remote origin --push
fi
printf '\n'
gh repo view "$FULL" --json nameWithOwner,url,isPrivate,defaultBranchRef
printf '%s\n' 'Teacher file: submission/portfolio_round1.py. Check reports/verification.json before submission.'
