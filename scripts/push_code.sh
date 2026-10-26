#!/usr/bin/env bash
# Deploy the committed code on the hub (reg1) to every worker over SSH. The workers
# have no GitHub credentials, so reg1 is their only source of code. Run on reg1 after
# committing; uncommitted changes are not sent. Safe to re-run.
#
#   bash scripts/push_code.sh
#
# Each worker gets a checkout at the same absolute path as on reg1 (so .venv and the
# editable install resolve identically) plus a ~/reg symlink to it. The checkout is a
# deployment mirror: never edit it on the worker, or the next push is refused.

set -u

REPO=$(cd "$(dirname "$0")/.." && pwd -P)
. "$REPO/env.sh"
BRANCH=$(git -C "$REPO" rev-parse --abbrev-ref HEAD)

FAIL=0
for h in $REG_WORKERS; do
  echo "== $(hostname -s) -> $h : $REPO ($BRANCH)"
  ssh -o BatchMode=yes -o ConnectTimeout=10 "$h" "
    set -e
    [ -d '$REPO/.git' ] || { mkdir -p '$REPO' && git init -q -b '$BRANCH' '$REPO'; }
    git -C '$REPO' config receive.denyCurrentBranch updateInstead
    [ -e ~/reg ] || ln -s '$REPO' ~/reg
  " && git -C "$REPO" push -q --force "$h:$REPO" "$BRANCH:$BRANCH" || { echo "!! push to $h failed"; FAIL=1; }
done
exit $FAIL
