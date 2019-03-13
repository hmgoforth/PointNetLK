#! /usr/bin/bash

# for output
OUTDIR=${HOME}/results/ex1/plk
mkdir -p ${OUTDIR}

# Python3 command
PY3="nice -n 10 python"

# categories for testing and the trained model
MODEL=${HOME}/results/ex1_pointlk_0915_model_best.pth
CMN="-i /home/yasuhiro/work/pointnet/ModelNet40 -c ./sampledata/modelnet40_half1.txt --format wt --pretrained ${MODEL}"

# perturbations
PERDIR=${HOME}/results/ex1/gt

# test PointNet-LK with given perturbations (see. 'ex1_genrot.sh' for perturbations)
${PY3} test_pointlk.py ${CMN} -o ${OUTDIR}/result_010.csv -p ${PERDIR}/pert_010.csv -l ${OUTDIR}/log_010.log
${PY3} test_pointlk.py ${CMN} -o ${OUTDIR}/result_000.csv -p ${PERDIR}/pert_000.csv -l ${OUTDIR}/log_000.log
${PY3} test_pointlk.py ${CMN} -o ${OUTDIR}/result_090.csv -p ${PERDIR}/pert_090.csv -l ${OUTDIR}/log_090.log
${PY3} test_pointlk.py ${CMN} -o ${OUTDIR}/result_060.csv -p ${PERDIR}/pert_060.csv -l ${OUTDIR}/log_060.log
${PY3} test_pointlk.py ${CMN} -o ${OUTDIR}/result_120.csv -p ${PERDIR}/pert_120.csv -l ${OUTDIR}/log_120.log
${PY3} test_pointlk.py ${CMN} -o ${OUTDIR}/result_030.csv -p ${PERDIR}/pert_030.csv -l ${OUTDIR}/log_030.log
${PY3} test_pointlk.py ${CMN} -o ${OUTDIR}/result_180.csv -p ${PERDIR}/pert_180.csv -l ${OUTDIR}/log_180.log
${PY3} test_pointlk.py ${CMN} -o ${OUTDIR}/result_020.csv -p ${PERDIR}/pert_020.csv -l ${OUTDIR}/log_020.log
${PY3} test_pointlk.py ${CMN} -o ${OUTDIR}/result_040.csv -p ${PERDIR}/pert_040.csv -l ${OUTDIR}/log_040.log
${PY3} test_pointlk.py ${CMN} -o ${OUTDIR}/result_050.csv -p ${PERDIR}/pert_050.csv -l ${OUTDIR}/log_050.log
${PY3} test_pointlk.py ${CMN} -o ${OUTDIR}/result_070.csv -p ${PERDIR}/pert_070.csv -l ${OUTDIR}/log_070.log
${PY3} test_pointlk.py ${CMN} -o ${OUTDIR}/result_080.csv -p ${PERDIR}/pert_080.csv -l ${OUTDIR}/log_080.log
${PY3} test_pointlk.py ${CMN} -o ${OUTDIR}/result_100.csv -p ${PERDIR}/pert_100.csv -l ${OUTDIR}/log_100.log
${PY3} test_pointlk.py ${CMN} -o ${OUTDIR}/result_110.csv -p ${PERDIR}/pert_110.csv -l ${OUTDIR}/log_110.log
${PY3} test_pointlk.py ${CMN} -o ${OUTDIR}/result_130.csv -p ${PERDIR}/pert_130.csv -l ${OUTDIR}/log_130.log
${PY3} test_pointlk.py ${CMN} -o ${OUTDIR}/result_140.csv -p ${PERDIR}/pert_140.csv -l ${OUTDIR}/log_140.log
${PY3} test_pointlk.py ${CMN} -o ${OUTDIR}/result_150.csv -p ${PERDIR}/pert_150.csv -l ${OUTDIR}/log_150.log
${PY3} test_pointlk.py ${CMN} -o ${OUTDIR}/result_160.csv -p ${PERDIR}/pert_160.csv -l ${OUTDIR}/log_160.log
${PY3} test_pointlk.py ${CMN} -o ${OUTDIR}/result_170.csv -p ${PERDIR}/pert_170.csv -l ${OUTDIR}/log_170.log


#EOF
