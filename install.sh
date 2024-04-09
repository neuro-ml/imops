# install from any folder
scriptDir=$(dirname -- "$(readlink -f -- "${BASH_SOURCE[0]}")")
cd "$scriptDir" || exit 1
rm imops/cpp/cpp_modules.cpython-311-x86_64-linux-gnu.so
pip install -e .
