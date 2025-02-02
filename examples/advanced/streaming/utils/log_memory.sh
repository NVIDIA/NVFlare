#!/bin/bash -e

echo "      date     time $(free -m | grep total | sed -E 's/^    (.*)/\1/g')"
counter=1
while [ $counter -le 400 ]; do
    echo "$(date '+%Y-%m-%d %H:%M:%S') $(free -m | grep Mem: | sed 's/Mem://g')"
    sleep 0.5
    ((counter++))
done
