# Tracker
Water skiing tracker

* Psaudo code for this tracker is in mainLogic.txt
* Most of the job is done by the Detector i.e YOLOv3, taken from https://github.com/cfotache/pytorch_objectdetecttrack (pyThorch)
and a tracker - CSRT from openCV.



Operating Instructions:

0. Clone this Repo 
1. Run the download_weights.sh script in the config folder.
   - you will need wget. if you dont have it, download the .exe from here- https://eternallybored.org/misc/wget/, and move it to a place in your PATH.
2. Create conda virtual env using environment.yml file
3. Put your video in a subDir videos 
4. Set your python interperter to this conda env and run ski_tracker.py
