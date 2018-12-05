#!/usr/bin/env bash
# Create 3 scp files mix, s1, s2 pulling random wavs
# usage:
# ./mk_scp.sh <type> <scp_name> <base-dir> <file-count>
# ./mk_scp.sh tr rel10 2spk8kmax 5

dir=$1
scp_name=$2
base_dir=$3
cnt=$4

shuf() { perl -MList::Util=shuffle -e 'print shuffle(<>);' "$@"; }

# exec basename is just too slow!
# find mix -type f -exec basename {} \;
# ls -1 "$dir/mix" | shuf | head -n "$cnt" > "${dir}_file$cnt.txt"

for file in $(ls -1 "$dir/mix" | shuf | head -n "$cnt");
do
  for tp in mix s1 s2;
  do
    echo "${file%.*} $base_dir/$dir/$tp/$file" >> "${scp_name}_${dir}_${tp}.scp"
  done
done

