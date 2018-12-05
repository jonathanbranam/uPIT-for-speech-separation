#!/usr/bin/env bash

dir=$1
cnt=$2

shuf() { perl -MList::Util=shuffle -e 'print shuffle(<>);' "$@"; }

# exec basename is just too slow!
# find mix -type f -exec basename {} \;
ls -1 "$dir/mix" | shuf | head -n "$cnt" > "${dir}_file$cnt.txt"

