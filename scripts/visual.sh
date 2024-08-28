#!/bin/bash

PYTHON_CMD="python3"
DIR="." 
CMD="export PYTHONPATH=\${PYTHONPATH}:$DIR"
$CMD

CMD="$PYTHON_CMD xxxx/yyyy/core/io/HTMLWriter.py $1 $2"
echo $CMD
$CMD

