# convert pipeline_overview.md to pipeline_overview.pdf
# requires pandoc and xelatex

set -e

cd "$(dirname "$0")"

# xelatex: the .md contains Unicode (·, p̃)
# DejaVu fonts: the default font has no ≈
pandoc pipeline_overview.md -o pipeline_overview.pdf \
    --pdf-engine=xelatex \
    -V geometry:margin=2cm \
    -V mainfont="DejaVu Serif" \
    -V monofont="DejaVu Sans Mono" \
    -M title="Pipeline overview"
