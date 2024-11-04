cd abides-core
python3 setup.py install
cd ../abides-markets
python3 setup.py install
cd tests
pytest -o log_cli=true --log-level INFO 