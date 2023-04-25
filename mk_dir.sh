#!/bin/bash



names=($(seq $1 $2 $3))

for name in "${names[@]}"
do
  mkdir "$4$name"
done

