#!/bin/bash

python ir_card_reg.py
if [ $? -ne 0 ]; then
  echo "Error in script1. Exiting."
  exit 1
fi

python y_intervolume_reg.py 
if [ $? -ne 0 ]; then
  echo "Error in script2. Exiting."
  exit 1
fi

python x_intervolume_reg.py 
if [ $? -ne 0 ]; then
  echo "Error in script3. Exiting."
  exit 1
fi

echo "Pipeline completed successfully."