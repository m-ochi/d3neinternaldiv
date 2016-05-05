#!/bin/sh

git submodule add git@bitbucket.org:m-ochi/ptebase.git
git submodule add git@github.com:tanakh/cmdline.git
git submodule add git@github.com:m-ochi/aliasmethod.git
git submodule init
git submodule update

ln -s ptebase pteself

cd ptebase/src/
git submodule add git@github.com:tanakh/cmdline.git
git submodule add git@github.com:m-ochi/aliasmethod.git
git submodule init
git submodule update
cd ../../

#git submodule update --init --recursive

cmake .
make clean
make

./train_d3ne \
  --train_ww=../dataminimal/st-st.csv \
  --train_wd=../dataminimal/st-corp.csv \
  --train_wl=../dataminimal/st-imp.csv \
  --output=vec_2nd_wo_norm.txt \
  --binary=0 \
  --size=2 \
  --order=2 \
  --negative=5 \
  --samples=1 \
  --threads=1 \
  --rho=0.05

#  --train_ww=../dataminimal/st-st.csv \
#  --train_wd=../dataminimal/st-corp.csv \
#  --train_wl=../dataminimal/st-imp.csv \

#./train_pte \
#  --train_ww=../data/st-st.csv \
#  --train_wd=../data/st-corp.csv \
#  --train_wl=../data/st-imp.csv \
#  --output=vec_2nd_wo_norm.txt
