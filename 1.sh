#!/bin/bash

for((i=0;i<=4;i+=1));
do
python main.py --temp_test_offset=$i;
done