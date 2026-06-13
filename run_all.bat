@echo off
cd /d "%~dp0"
@REM 替换Tensorflow为你的conda环境名称
call conda activate xwtj

pushd "%~dp0Recall"
call python Recall_itemcf.py
call python DSSM_recall.py
call python Recall_merge.py
popd

pushd "%~dp0Rank"
call python Feat_Eng.py
call python din_rank.py
call python DCN_rank.py
popd

pause