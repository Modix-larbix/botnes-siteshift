#!/bin/sh
# Download the Franka Emika Panda model the simulator loads, from MuJoCo Menagerie at a pinned commit, and check it.
#
#   sh assets/fetch_panda.sh
#
# It writes menagerie/franka_emika_panda/ next to siteagent/ (where siteagent/world/build.py looks for panda.xml):
# panda.xml, the meshes panda.xml references, and the model's LICENSE (Apache-2.0) and README.md. It copies nothing
# else, then checks every file against assets/PANDA_ASSETS.sha256 and exits non-zero on any difference.
# The model is not part of this repository: it belongs to the MuJoCo Menagerie project (Apache-2.0).
# Needs git and network access to github.com, once. The simulator itself never uses the network.
set -eu

REPO_URL="https://github.com/google-deepmind/mujoco_menagerie.git"
COMMIT="71f066ad0be9cd271f7ed58c030243ef157af9f4"

HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(dirname "$HERE")
LIST="$HERE/PANDA_ASSETS.sha256"
TARGET="$ROOT/menagerie/franka_emika_panda"

check() {
    (cd "$TARGET" && grep -v '^#' "$LIST" | shasum -a 256 -c --quiet -)
}

if [ -d "$TARGET" ]; then
    if check; then
        echo "Panda model already in place and verified: $(grep -vc '^#' "$LIST") files"
        exit 0
    fi
    echo "refused: $TARGET exists but differs from the pinned files; move it away and run this script again" >&2
    exit 1
fi

WORK=$(mktemp -d "${TMPDIR:-/tmp}/siteshift_panda.XXXXXX")
trap 'rm -rf "$WORK"' EXIT
git clone --quiet --filter=blob:none --sparse --no-checkout "$REPO_URL" "$WORK/menagerie"
git -C "$WORK/menagerie" sparse-checkout set franka_emika_panda
git -C "$WORK/menagerie" -c advice.detachedHead=false checkout --quiet "$COMMIT"
test "$(git -C "$WORK/menagerie" rev-parse HEAD)" = "$COMMIT"

mkdir -p "$TARGET/assets"
grep -v '^#' "$LIST" | while read -r _sha rel; do
    cp "$WORK/menagerie/franka_emika_panda/$rel" "$TARGET/$rel"
done

if check; then
    echo "Panda model fetched from mujoco_menagerie $COMMIT and verified: $(grep -vc '^#' "$LIST") files"
else
    echo "error: the fetched files differ from assets/PANDA_ASSETS.sha256" >&2
    exit 1
fi
