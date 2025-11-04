**NOTE**  
Test on sign detection ok, sign classification also ok!  
Next to do: hyperparam tuning; further experiments

**TO USE THIS**  
1. install ultralytics
2. run ```make_yaml_dataset.ipynb```
3. try playground.

**Reminder**  
For the time being, in order to switch task (from is_sign to charname or transliteration),  
you should change the configuration in ```make_yaml_dataset.ipynb``` and overwrite the YOLO dataset.  
TODO : generate all 3 types of annotation side by side.