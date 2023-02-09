python3.11 setup.py bdist_wheel
rm -rf .venv-test
python3.11 -m venv .venv-test
source .venv-test/bin/activate
.venv-test/bin/pip install --upgrade pip
.venv-test/bin/pip install 'dist/nanopyx-0.0.1-cp311-cp311-macosx_10_9_universal2.whl[test]'
cd ..
NanoPyx/.venv-test/bin/pytest NanoPyx/tests
