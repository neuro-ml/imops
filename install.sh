# install from any folder
scriptDir=$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")
cd "$scriptDir" || exit 1
cd imops/cpp
rm -rf build
mkdir build
cd build
cmake ..
make
cd "$scriptDir" || exit 1
pip install -e .
