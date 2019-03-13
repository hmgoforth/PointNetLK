#! /usr/bin/bash

# generate perturbations for each object (for each 'ModelNet40/[category]/test/*')

# for output
OUTDIR=${HOME}/results/gt
mkdir -p ${OUTDIR}

# Python3 command
PY3="nice -n 10 python"

# categories for testing
CMN="-i /home/yasuhiro/work/pointnet/ModelNet40 -c ./sampledata/modelnet40_half1.txt --no-translation"

${PY3} generate_perturbations.py ${CMN} -o ${OUTDIR}/pert_00.csv --mag 0.0
${PY3} generate_perturbations.py ${CMN} -o ${OUTDIR}/pert_01.csv --mag 0.1
${PY3} generate_perturbations.py ${CMN} -o ${OUTDIR}/pert_02.csv --mag 0.2
${PY3} generate_perturbations.py ${CMN} -o ${OUTDIR}/pert_03.csv --mag 0.3
${PY3} generate_perturbations.py ${CMN} -o ${OUTDIR}/pert_04.csv --mag 0.4
${PY3} generate_perturbations.py ${CMN} -o ${OUTDIR}/pert_05.csv --mag 0.5
${PY3} generate_perturbations.py ${CMN} -o ${OUTDIR}/pert_06.csv --mag 0.6
${PY3} generate_perturbations.py ${CMN} -o ${OUTDIR}/pert_07.csv --mag 0.7
${PY3} generate_perturbations.py ${CMN} -o ${OUTDIR}/pert_08.csv --mag 0.8
${PY3} generate_perturbations.py ${CMN} -o ${OUTDIR}/pert_09.csv --mag 0.9
${PY3} generate_perturbations.py ${CMN} -o ${OUTDIR}/pert_10.csv --mag 1.0
${PY3} generate_perturbations.py ${CMN} -o ${OUTDIR}/pert_11.csv --mag 1.1
${PY3} generate_perturbations.py ${CMN} -o ${OUTDIR}/pert_12.csv --mag 1.2
${PY3} generate_perturbations.py ${CMN} -o ${OUTDIR}/pert_13.csv --mag 1.3
${PY3} generate_perturbations.py ${CMN} -o ${OUTDIR}/pert_14.csv --mag 1.4
${PY3} generate_perturbations.py ${CMN} -o ${OUTDIR}/pert_15.csv --mag 1.5


#EOF
