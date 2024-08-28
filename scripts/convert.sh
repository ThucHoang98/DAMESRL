#!/bin/bash



PYTHON_CMD="python3"
DIR="." 
CMD="export PYTHONPATH=\${PYTHONPATH}:$DIR"
$CMD


if [ "$1" = "05" ]
then
CMD="$PYTHON_CMD liir/dame/core/io/CoNLL2005Reader.py $2 $3"
echo $CMD
$CMD

fi

if [ "$1" = "12" ]
then
CMD="$PYTHON_CMD liir/dame/core/io/CoNLL2012Reader.py $2 $3"
echo $CMD
$CMD

fi

