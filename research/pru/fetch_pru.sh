#!/bin/sh
# EUROCONTROL performance data (ansperformance.eu, served from eurocontrol.int) used by the pipeline:
#
#   taxi_out_additional_time_{2025,2026}.csv  monthly taxi-out statistics per airport
#       -> research/final/level.py (airport-month level calibration)
#   all_pre_departure_delays_{2025,2026}.csv  daily pre-departure delay per airport
#       -> research/final/levers/daily.py (daily reconciliation layer, switchable)
#
# The files are not redistributed in this repository (see docs/DATA.md for the terms). This script
# downloads them into research/pru/ and checks each against the sha256 of the copy the submission
# was built from. The publisher updates these files in place; a hash mismatch means the server
# copy changed after 2026-10-01, not that the download failed.
set -e
cd "$(dirname "$0")"
BASE="https://www.eurocontrol.int/performance/data/download/csv"
for f in taxi_out_additional_time_2025 taxi_out_additional_time_2026 \
         all_pre_departure_delays_2025 all_pre_departure_delays_2026; do
  curl -fsSL -o "$f.csv" "$BASE/$f.csv"
done
cat > expected.sha256 <<'EOF'
4aedf19520d3f84bebc418b56835762c907b714a9c19e751bb8658f9e62ce598  taxi_out_additional_time_2025.csv
b72040e0987e6c7d5706279d645ef9f814a6506165c83f89d8509022154a1076  taxi_out_additional_time_2026.csv
d742eeb626e5f9f7e0c756e2ae3cf90078f4627e989a2e31b8ed21ba8f5510be  all_pre_departure_delays_2025.csv
5952d20ef5f9c4245018dd6986064c5c4aa15d568a6c0c2dcdcbc73bf8a43fd2  all_pre_departure_delays_2026.csv
EOF
if command -v sha256sum >/dev/null 2>&1; then sha256sum -c expected.sha256; else shasum -a 256 -c expected.sha256; fi
