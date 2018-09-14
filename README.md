<!--<a href="https://giphy.com/gifs/U7MUyyxzyaKoDVdZ9V"> <img  src="https://media.giphy.com/media/U7MUyyxzyaKoDVdZ9V/giphy.gif" title="Overly Ambitious Title"/></a>-->
![Imgur](https://i.imgur.com/czOS770.png)

###  [Introduction](#introduction-1)
###  [Prerequisites](#prerequisites-1)
###  [Resources](#resources-1)
###  [Detection](#detection-implementation)
  *  [Minizeption](#minizeption)
###  [Usage](#usage-1)
###  [Demo Kind Of](#demo)


## Introduction
This is a hobby project to explore the field of computer vision abit. The main challenge is to classify differences
in a scene. Example, different people being present in a room with a static background.
<br/>
#### Constraints
* No pre-trained model can be used.
* All training has to be done on data sampled by the camera.
* The objective is to be able to do classification on individual images not sequences, eliminating time-dependant models.

A tiny tool is built around this to make it easier to try it out.

## Prerequisites
  * camera compatible with opencv2 
  * python 3.x

#### Resources
  * https://www.deeplearningbook.org/contents/convnets.html (What Convolutions is)
  * https://arxiv.org/pdf/1803.08834.pdf (Good Summary of DL in computer vision)
  * https://arxiv.org/pdf/1409.4842.pdf (GoogLeNet) (Inception Modules)
  * https://arxiv.org/pdf/1711.08132.pdf (LSNNs) 
  * https://arxiv.org/pdf/1502.03167.pdf (Batch Norm, Faster Training)
  * https://arxiv.org/pdf/1502.01852.pdf (Proper Initialization, Faster Training)
  * https://arxiv.org/pdf/1706.05350.pdf (L2 + BatchNorm = :( )
  * https://arxiv.org/pdf/1612.01490.pdf (Regularization With Gaussian Noise, Seems Cool)

## Detection Implementation
- [x] [Inception Module](#minizeption)
- [ ] LSNN Module

### Minizeption
Structure of the minizeption network
![Imgur](https://i.imgur.com/vyyhEyj.png)


<br/>
<br/>
<br/>
<br/>

## Usage

> Clone repo and install dependencies

```bash

> git clone https://github.com/JonasRSV/See-A-Stuff.git
> cd See-A-Stuff
> pip3 install -r requirements.txt
> python3 see-a-thing -h

                                                                                                                                
[92m         _______. _______  _______           ___           .___________. __    __   __  .__   __.   _______     __       ___      [0m
[92m        /       ||   ____||   ____|         /   \          |           ||  |  |  | |  | |  \ |  |  /  _____|   /_ |     / _ \   [0m
[92m       |   (----`|  |__   |  |__    ______ /  ^  \   ______`---|  |----`|  |__|  | |  | |   \|  | |  |  __      | |    | | | |  [0m
[92m        \   \    |   __|  |   __|  |______/  /_\  \ |______|   |  |     |   __   | |  | |  . `  | |  | |_ |     | |    | | | |  [0m
[92m    .----)   |   |  |____ |  |____       /  _____  \           |  |     |  |  |  | |  | |  |\   | |  |__| |     | |  __| |_| |  [0m
[92m    |_______/    |_______||_______|     /__/     \__\          |__|     |__|  |__| |__| |__| \__|  \______|     |_| (__)\___/   [0m
                                                                                                                                
usage: see-a-thing [-h] [--run {record,train}]
                   [--serve {commandline,websocket}] [--clean] [--label LABEL]
                   [-f MAX_FREQUENCY] [-t TIME] [-d] [--overwrite] [--fix]
                   [--training_path TRAINING_PATH] [--model_path MODEL_PATH]
                   [--data]

optional arguments:
  -h, --help            show this help message and exit
  --run {record,train}, -r {record,train}
                        The recorder records the data used for training The
                        train creates the model that get served.
  --serve {commandline,websocket}, -s {commandline,websocket}
  --clean               Clean the model and training directory
  --label LABEL         Label for the recorded data
  -f MAX_FREQUENCY, --max_frequency MAX_FREQUENCY
                        Max Frequency camera will run at in Hz
  -t TIME, --time TIME  Time the camera should run for (In seconds)
  -d, --display         Display what is being recorded
  --overwrite           Will overwrite existing model when combined with train
  --fix                 'Fix all prequisites for me' Use with caution, will
                        remove training data.
  --training_path TRAINING_PATH
                        (Optional) Path to the training directory
  --model_path MODEL_PATH
                        (Optional) Path to the model directory
  --data                Print Available Data


```

> Record some things

```bash

> python3 see-a-thing -r record --label thing1 -f 10 -t 120
> python3 see-a-thing -r record --label thing2 -f 10 -t 120
> python3 see-a-thing -r record --label thing3 -f 10 -t 120

Add more data for thing1 if you want

> python3 see-a-thing -r record --label thing1 -f 10 -t 120

```

> Train the model

```bash

> python3 see-a-thing -r train

```

> Serve the model

```bash

> python3 see-a-thing -s commandline (For commandline GUI)
> python3 see-a-thing -s websocket (For starting a socket serving predictions at port 5000)

```

<br/>
<br/>
<br/>

## Demo


### Training Period
* 100 seconds of 10 hz recording of empty background
* 100 seconds of 10 hz recording of me walking around in camera view
* 50 epochs 

<br/>

![Imgur](https://i.imgur.com/XTLvcci.png)

<br/>

* CLI UI showing results when i'm walkin in and out of the camera view at different positions

<br/>

<a> <img width=800px src="demo.gif" title="demo"/></a>
