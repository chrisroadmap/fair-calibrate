#!/usr/bin/env bash

script_dir=$PWD

cd ../../../../../data/cmip6
bash wget-20230216102749.sh -s
bash wget-20230216132736.sh -s
bash wget-20230216182759.sh -s
bash wget-20230216192744.sh -s
bash wget-20230216192753.sh -s
bash wget-20230216192803.sh -s

cd ${script_dir}
