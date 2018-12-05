#!/usr/bin/env bash

dir=$1
filenm=$2
dst=$3

for d in mix s1 s2;
do
  mkdir -p "$dst/$dir/$d"
done

for file in $(<"$filenm");
do
  for d in mix s1 s2;
  do
    cp "$dir/$d/$file" "$dst/$dir/$d"
  done
done
