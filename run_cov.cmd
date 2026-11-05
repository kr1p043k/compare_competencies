@echo off
cd /d "C:\Users\291A~1\AppData\Local\Temp\opencode\wt-night2"
set SECRET_KEY=8f3a1c9e5b7d2f4a6c8e0b1d3f5a7c9e
set PYTHONIOENCODING=utf-8
python -m pytest tests -q --ignore=tests/load --cov=src --cov-report=term -p no:cacheprovider > cov.out.log 2> cov.err.log
