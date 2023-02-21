#!/usr/bin/env bash

script_dir=$PWD

cd ../../../../../data/cmip6
for file in *.sh
do
    bash $file -s
done

cd ${script_dir}
