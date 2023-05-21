Simple inference code for SynthIA. To run on your own days and times you can copy the format of the example `2015.11.timelist.txt` file in the timelists folder.

Step 1: 

fetch the models from this URL: 

Step 2:

`pip install -r requirements.txt` -- make sure you install a version of PyTorch compatible with a GPU: https://pytorch.org/

Step 3:

`python synthia_inference.py --timelist ./timelists/2015.11.timelist.txt`
