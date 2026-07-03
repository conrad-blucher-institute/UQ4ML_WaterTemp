@echo off
rem Bare `esb` launcher (Windows). Works from the repo root, or from anywhere
rem if you add this repo folder to PATH. Uses whatever python is active
rem (activate your conda env first).
python -m esb %*
