#!/usr/bin/env bash
for id in 1 2 3 4 5 6 7 8
do
    ./site-$id/startup/start.sh site-$id localhost &
done

