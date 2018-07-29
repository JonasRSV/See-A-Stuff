### See-A-Stuff

A hobby project playing with opencv and some image recognition. The objective is to distinguish between individuals
with a fixed background given that they've been recorded for a short while.


The project is very much in progress.


To try it out. 
  1. Buy a Camera that can be connected to your computer
  2. Pull this Repo
  3. Run: pip3 install -r requirements.txt
  4. Place the camera in whichever position you want it. It is important that the background is static.
  5. Run: python3 stuffseer.py -record
  6. Let walk around just as one would normally do infront of the camera for a while until the program exits.
  7. Repeat 6 with as many people as you want, preferably with the same people during different times of the day with different clothes and ligthing for better results.
  8. Run: python3 stuffseer -train
  9. Run: python3 stuffseer -see (This has not been finished yet)

