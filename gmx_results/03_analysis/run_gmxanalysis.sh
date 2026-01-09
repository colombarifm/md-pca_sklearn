#!/bin/bash

trjconv="gmx_mpi trjconv -s ../02_nvt/nvt_run.tpr -n ../files/index.ndx"

echo 2 0 | ${trjconv} -f ../02_nvt/nvt_run.xtc -pbc mol -center -o trj_center.xtc
echo 2 0 | ${trjconv} -f trj_center.xtc -fit rot+trans -o trj_fit.xtc
echo 2 0 | ${trjconv} -f trj_fit.xtc -fit rot+trans -o nvt.pdb -dump 0

rm trj_center.xtc

echo 2 | gmx_mpi rmsf -f trj_fit.xtc -s ../02_nvt/nvt_run.tpr  -nofit -xvg none -ox gmx_avg_structure.pdb -o gmx_rmsf.xvg

cat gmx_rmsf.xvg | awk '{printf "%7.4f\n", $2}' > gmx_rmsf_noindex.xvg

echo 2 2 | gmx_mpi covar -s ../02_nvt/nvt_run.tpr -f trj_fit.xtc -n ../files/index.ndx \
  -o gmx_eigenvalues.xvg -v gmx_eigenvectors.trr -ascii gmx_covariancemtx.dat -xpm gmx_covariancemtx.xpm

echo 2 2 | gmx_mpi anaeig -f trj_fit.xtc -s ../02_nvt/nvt_run.tpr -v gmx_eigenvectors.trr -n ../files/index.ndx -eig gmx_eigenvalues.xvg \
  -extr gmx_pseudotraj_pc.pdb -first 1 -last 3 -nframes 31 -max 0.1 -proj gmx_projection_pc.xvg -split

csplit -z gmx_projection_pc.xvg /"&"/ '{*}' -f gmx_projection_clean_pc --suppress-matched

sed -i "\;@;d" gmx_projection_clean_pc00
sed -i "\;@;d" gmx_projection_clean_pc01
sed -i "\;@;d" gmx_projection_clean_pc02

printf "%6s\t%9s\t%9s\t%9s\n" "#t" "pc1" "pc2" "pc3" > all_pcs.dat
paste gmx_projection_clean_pc00 gmx_projection_clean_pc01 gmx_projection_clean_pc02 | awk '{printf "%6.1f\t%9.6f\t%9.6f\t%9.6f\n", $1, $2, $4, $6}' >> all_pcs.dat


gmx_mpi xpm2ps -f gmx_covariancemtx.xpm -o gmx_covariancemtx.eps

convert -background white -flatten -colorspace sRGB -density 600 gmx_covariancemtx.eps -quality 90 gmx_covariancemtx.png
