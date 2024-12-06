# -----------------------------------------------------------------------------
# This is a script to test python source code.
#
# Command:
#   source pytest_all.sh
# -----------------------------------------------------------------------------
#!/bin/bash

fpath=test_dataset.py

pytest -vv $fpath
pytest -vv --ignore $fpath

dpaths=(
    data/mnist
    data/fashion_mnist
    data/cifar10
    data/cifar100
)
for dpath in ${dpaths[@]}; do
    rm -r $dpath
done

unset fpath
unset dpaths
unset dpath