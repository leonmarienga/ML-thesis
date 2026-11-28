#!/usr/bin/env bash
set -euo pipefail
dest=${1:?provide a new output directory}
mkdir "$dest"
for y in 2012 2018 2024; do
 (
  root=https://mesonet.agron.iastate.edu
  curl -fsS --retry 1 --max-time 30 --max-filesize 12000000 "$root/cgi-bin/request/gis/outlooks.py?year1=$y&month1=5&day1=1&hour1=0&minute1=0&year2=$y&month2=5&day2=8&hour2=0&minute2=0&type=C&d=1&geom=geom" -o "$dest/$y.zip"
  curl -fsS --retry 1 --max-time 30 --max-filesize 1000000 "$root/cgi-bin/afos/retrieve.py?pil=SWODY1&fmt=text&sdate=$y-05-01&edate=$y-05-02&limit=25" -o "$dest/$y.txt"
 ) &
 pids[$y]=$!
done
for y in 2012 2018 2024; do wait "${pids[$y]}"; done
