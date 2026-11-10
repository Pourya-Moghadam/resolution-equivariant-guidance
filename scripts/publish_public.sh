#!/usr/bin/env bash
# Publish the public mirror of this repository: a filtered copy of the whole history, force-pushed to the personal
# GitHub repository. Run on reg1 after every `git push origin main`:
#
#   git push origin main && bash scripts/publish_public.sh && bash scripts/push_code.sh
#
# The filter drops the private notes listed in $REG_DATA/public_filter_paths.txt and applies the replacement list
# $REG_DATA/public_filter_rules.txt (git filter-repo syntax) to every file version and commit message; both files
# live outside the repository so the mirror never carries them. The push is refused if any string from
# $REG_DATA/public_filter_check.txt survives anywhere in the filtered history (case-insensitive). The filter is
# deterministic, so unchanged history maps to the same rewritten commits and the force-push is a fast-forward.
set -eu
REPO=$(cd "$(dirname "$0")/.." && pwd -P)
. "$REPO/env.sh"
PUBLIC=${REG_PUBLIC_REMOTE:-https://github.com/Pourya-Moghadam/resolution-equivariant-guidance.git}
PATHS=$REG_DATA/public_filter_paths.txt
RULES=$REG_DATA/public_filter_rules.txt
CHECK=$REG_DATA/public_filter_check.txt
for f in "$PATHS" "$RULES" "$CHECK"; do [ -f "$f" ] || { echo "missing $f" >&2; exit 1; }; done
W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT
git clone -q --no-local --branch main "$REPO" "$W/src"
git -C "$W/src" filter-repo --quiet --invert-paths --paths-from-file "$PATHS" --replace-text "$RULES" --replace-message "$RULES"
HITS=$( { git -C "$W/src" log -p --all --format='%B%n%an %ae'; git -C "$W/src" ls-tree -r --name-only HEAD; } | grep -i -c -F -f "$CHECK" || true)
if [ "$HITS" != 0 ]; then echo "publish_public: $HITS forbidden strings remain in the filtered history; not pushed" >&2; exit 1; fi
N=$(git -C "$W/src" rev-list --count main)
git -C "$W/src" push -q --force "$PUBLIC" main
echo "public mirror updated: $N commits -> $PUBLIC"
