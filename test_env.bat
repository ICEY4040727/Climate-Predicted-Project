@echo off
echo Testing environment... > test_output.txt
C:\Users\ICEY\.conda\envs\climate312\python.exe E:\Climate-D-S\test_env.py >> test_output.txt 2>&1
echo Done! Check test_output.txt
type test_output.txt
pause

