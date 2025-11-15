cd python
rm dist/*.whl
pip install -r install_requirements.txt
#CTRANSLATE2_ROOT=/Users/masj/dev/CTranslate2/install python setup.py bdist_wheel
#CIBW_BUILD=cp313* python3 -m cibuildwheel . --output-dir dist --platform macos --archs $(python3 -c "import platform; print(platform.machine())") 

export CTRANSLATE2_ROOT=/Users/masj/dev/CTranslate2/install
# Install build tools
pip install --upgrade build delocate

# Build the wheel for your current Python
python -m build --wheel --outdir dist

# Bundle .dylibs on macOS
#delocate-wheel -w dist dist/*.whl
export DYLD_LIBRARY_PATH=$CTRANSLATE2_ROOT/lib:$DYLD_LIBRARY_PATH
delocate-wheel -w dist  dist/*.whl
pip install  --force-reinstall dist/*.whl
python ../ver.py