#!/bin/bash
set -e -u -x

yum install -y cmake

WHEELS_ROOT=$(realpath wheelhouse)
echo $WHEELS_ROOT

function repair_wheel {
    wheel="$1"
    if ! auditwheel show "$wheel"; then
        echo "Skipping non-platform wheel $wheel"
    else
        auditwheel repair "$wheel" --plat "$PLAT" -w $WHEELS_ROOT
    fi
}

# Update pips
for INTERNAL in /opt/_internal/*/bin/python; do
    if echo $INTERNAL | grep "cpython-3.10\|cpython-3.9\|cpython-3.8"; then 
        "${INTERNAL}" -m pip install --upgrade pip
    fi    
done

# Compile wheels
for PYBIN in /opt/python/*/bin; do
    if echo $PYBIN | grep "cp310\|cp39\|cp38"; then 
        "${PYBIN}/pip" install -r requirements.txt
        "${PYBIN}/pip" wheel . --no-deps -w $WHEELS_ROOT
    fi   
done

# Bundle external shared libraries into the wheels
for whl in $WHEELS_ROOT/*.whl; do
    repair_wheel "$whl"
done
