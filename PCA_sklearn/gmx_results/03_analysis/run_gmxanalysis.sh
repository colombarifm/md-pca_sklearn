#!/bin/bash

trjconv="gmx_mpi trjconv -s ../02_nvt/nvt.tpr -n ../files/index.ndx"

echo 2 0 | ${trjconv} -f ../02_nvt/nvt.xtc -pbc mol -center -o trj_center.xtc
echo 2 0 | ${trjconv} -f trj_center.xtc -fit rot+trans -o trj_fit.xtc
echo 2 0 | ${trjconv} -f trj_fit.xtc -fit rot+trans -o nvt.pdb -dump 0

rm trj_center.xtc

echo 2 | gmx_mpi rmsf -f trj_fit.xtc -s ../02_nvt/nvt.tpr  -nofit -xvg none -ox gmx_avg_structure.pdb -o gmx_rmsf.xvg

cat gmx_rmsf.xvg | awk '{printf $2}' > gmx_rmsf_noindex.xvg

echo 2 2 | gmx_mpi covar -s ../02_nvt/nvt.tpr -f trj_fit.xtc -n ../files/index.ndx \
  -o gmx_eigenvalues.xvg -v gmx_eigenvectors.trr -ascii gmx_covariancemtx.dat -xpm gmx_covariancemtx.xpm

echo 2 2 | gmx_mpi anaeig -f trj_fit.xtc -s ../02_nvt/nvt.tpr -v gmx_eigenvectors.trr -n ../files/index.ndx -eig gmx_eigenvalues.xvg \
  -extr gmx_pseudotraj_pc.pdb -first 1 -last 3 -nframes 31 -max 0.1

gmx_mpi xpm2ps -f gmx_covariancemtx.xpm -o gmx_covariancemtx.eps
