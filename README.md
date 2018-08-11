# See-A-Stuff

###  [Introduction](#introduction)
###  [Prequisites](#prequisites)
###  [Detection](#detection-implementation)
###  [Try it out](#trying-it-out)


## Introduction
This is a hobby project to explore the field of computer vision abit. The main challenge is to differentiate between
individuals in a room with a static background. 

<br/>

A tiny tool is built around this to make it easier to try it out.

### Prequisites
  * camera compatible with opencv2 
  * python 3.x

### Detection Implementation
... In progress

### Trying it out

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

