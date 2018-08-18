<!--<a href="https://giphy.com/gifs/U7MUyyxzyaKoDVdZ9V"> <img  src="https://media.giphy.com/media/U7MUyyxzyaKoDVdZ9V/giphy.gif" title="Overly Ambitious Title"/></a>-->
![Imgur](https://i.imgur.com/czOS770.png)

###  [Introduction](#introduction)
###  [Prequisites](#prequisites)
###  [Resources](#resources)
###  [Detection](#detection-implementation)
  *  [Minizeption](#minizeption)
###  [Try it out](#trying-it-out)


## Introduction
This is a hobby project to explore the field of computer vision abit. The main challenge is to classify differences
in a scene. Example, different people being present in a room with a static background.
<br/>
#### Constraints
* No pre-trained model can be used.
* All training has to be done on data sampled by the camera.
* The objective is to be able to do classification on individual images not sequences, eliminating time-dependant models.

A tiny tool is built around this to make it easier to try it out.

## Prequisites
  * camera compatible with opencv2 
  * python 3.x

#### Resources
  * https://www.deeplearningbook.org/contents/convnets.html (What Convolutions is)
  * https://arxiv.org/pdf/1803.08834.pdf (Good Summary of DL in computer vision)
  * https://arxiv.org/pdf/1409.4842.pdf (GoogLeNet)
  * https://arxiv.org/pdf/1711.08132.pdf (LSNNs) 

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

## Trying it out

To try it out install all the requirements in the requirements.txt file using
```bash
pip3 install -r requirements.txt
```

Run the main file to see all available option

```bash
python3 see-a-thing -h
```

A minimal usecase would be.

#### Record the Background 
```bash
python3 see-a-thing -r record --label background -f 10 -t 120
```
#### Record X individuals
```bash
python3 see-a-thing -r record --label NAME -f 10 -t 120
```
> You can easily record additional data for a individual, just use the same label and the CLI will append to the current data it has.

#### Train the model
```bash
python3 see-a-thing -r train
```
#### Start a monitor
```bash
python3 see-a-thing -s commandline
```
> for a commandline gui

```bash
python3 see-a-thing -s websocket
```
> to serve the results from port 5000. The websocket might be nice for a GUI.

