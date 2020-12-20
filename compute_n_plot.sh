#!/bin/bash
cd $1
make clean
make
([ $? -eq 0 ] && echo "Compilation : success"; ./projet1; python3 tracer_solution.py &) || echo "Compilation: failed"
