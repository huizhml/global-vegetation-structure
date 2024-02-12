#!/bin/bash
for dir in ~/*/; do
    echo -n "$dir: "
    find "$dir" -type f | wc -l
done
