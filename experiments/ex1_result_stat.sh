#! /usr/bin/bash

# Python3 command
PY3="nice -n 10 python"

# for output
#RES="${HOME}/results/ex1/icp"
RES="${HOME}/results/ex1/plk"

# gather 'result_*.csv' to 'result.csv'
${PY3} result_stat.py --hdr > ${RES}/result.csv
${PY3} result_stat.py -i ${RES}/result_000.csv --val 0 >> ${RES}/result.csv
${PY3} result_stat.py -i ${RES}/result_010.csv --val 10 >> ${RES}/result.csv
${PY3} result_stat.py -i ${RES}/result_020.csv --val 20 >> ${RES}/result.csv
${PY3} result_stat.py -i ${RES}/result_030.csv --val 30 >> ${RES}/result.csv
${PY3} result_stat.py -i ${RES}/result_040.csv --val 40 >> ${RES}/result.csv
${PY3} result_stat.py -i ${RES}/result_050.csv --val 50 >> ${RES}/result.csv
${PY3} result_stat.py -i ${RES}/result_060.csv --val 60 >> ${RES}/result.csv
${PY3} result_stat.py -i ${RES}/result_070.csv --val 70 >> ${RES}/result.csv
${PY3} result_stat.py -i ${RES}/result_080.csv --val 80 >> ${RES}/result.csv
${PY3} result_stat.py -i ${RES}/result_090.csv --val 90 >> ${RES}/result.csv
${PY3} result_stat.py -i ${RES}/result_100.csv --val 100 >> ${RES}/result.csv
${PY3} result_stat.py -i ${RES}/result_110.csv --val 110 >> ${RES}/result.csv
${PY3} result_stat.py -i ${RES}/result_120.csv --val 120 >> ${RES}/result.csv
${PY3} result_stat.py -i ${RES}/result_130.csv --val 130 >> ${RES}/result.csv
${PY3} result_stat.py -i ${RES}/result_140.csv --val 140 >> ${RES}/result.csv
${PY3} result_stat.py -i ${RES}/result_150.csv --val 150 >> ${RES}/result.csv
${PY3} result_stat.py -i ${RES}/result_160.csv --val 160 >> ${RES}/result.csv
${PY3} result_stat.py -i ${RES}/result_170.csv --val 170 >> ${RES}/result.csv
${PY3} result_stat.py -i ${RES}/result_180.csv --val 180 >> ${RES}/result.csv

#EOF