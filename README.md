# SynthIA: a synthetic inversion approximation for the Stokes vector fusing SDO and Hinode into a virtual observatory

This is inference code that fetches IQUV recordings directly from DRMS then runs SynthIA on them for a number of targets. Stay tuned for a more exciting README soon.

```@article{higgins2022synthia,
  title={SynthIA: a synthetic inversion approximation for the Stokes vector fusing SDO and Hinode into a virtual observatory},
  author={Higgins, Richard EL and Fouhey, David F and Antiochos, Spiro K and Barnes, Graham and Cheung, Mark CM and Hoeksema, J Todd and Leka, KD and Liu, Yang and Schuck, Peter W and Gombosi, Tamas I},
  journal={The Astrophysical Journal Supplement Series},
  volume={259},
  number={1},
  pages={24},
  year={2022},
  publisher={IOP Publishing}
}
```

To run SynthIA is a few simple steps! If you want to use your own date/times, create a timelist file similar to the one we have,`2015.11.timelist.txt`, in the timelists folder. Your first install some requirements, then download some models, and finally run inference. Make sure you install a version of PyTorch compatible with a GPU: https://pytorch.org/

```pip install -r requirements.txt
wget https://www.dropbox.com/s/x4lrx2npy4zv403/models.zip?dl=0
unzip models.zip
python synthia_inference.py --timelist ./timelists/2015.11.timelist.txt
```

Note: uncomment out the rest of the targets in the target list in line 160 to generate more types of outputs.
