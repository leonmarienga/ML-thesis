#!/usr/bin/env bash
set -euo pipefail
output=${1:-public-clock-inputs}
mkdir -p "$output"
cd "$output"
for day in 28 31; do
 curl -fsS --max-time 45 "https://www.cnrfc.noaa.gov/archive2/2022/Dec/RVFverification/MHBC1.12${day}2022.12_rvf.png" -o "MHBC1-${day}.png"
done
curl -fsS --max-time 45 'https://mesonet.agron.iastate.edu/cgi-bin/afos/retrieve.py?pil=RVFSJ&fmt=text&sdate=2022-12-28&edate=2023-01-01&limit=40' -o iem-december.txt
curl -fsS --max-time 45 -G 'https://api.waterdata.usgs.gov/ogcapi/v0/collections/continuous/items' --data-urlencode monitoring_location_id=USGS-11335000 --data-urlencode parameter_code=00065 --data-urlencode datetime=2022-12-28T16:00:00Z/2023-01-05T13:00:00Z --data-urlencode limit=10000 --data-urlencode f=json -o usgs-stage-modern.json
