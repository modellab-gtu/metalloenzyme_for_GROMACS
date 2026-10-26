#!/bin/bash
# Download the ZAFF library (Peters et al., JCTC 6, 2935 (2010)) from the Amber MCPB.py/ZAFF tutorial.
set -e
cd "$(dirname "$(readlink -f "$0")")"
for f in ZAFF.prep ZAFF.frcmod; do
  python3 -c "import urllib.request,sys; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])" \
    "https://ambermd.org/tutorials/advanced/tutorial20/files/zaff/$f" "$f"
done
grep -q " INT " ZAFF.prep && echo "ZAFF files ready in $PWD"
