#!/bin/bash

set -e

pip uninstall libiq -y
rm -rf build dist src/libiq.egg-info src/libiq/_libiqwrapped.* src/libiq/libiqwrapped.py src/libiq_swig/libiq_wrapped_wrap.cpp
python setup.py build_ext --force
pip install -e . --no-cache-dir