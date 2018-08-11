# See-A-Stuff

[Prequisites][Prequisites]

[Try it out][Trying it out]

[Detection][Detection Implementation]

This is a hobby project to explore the field of computer vision abit. The main challenge is differentiate between
individuals in a room with a static background. Around this is a tiny CLI tool that wraps some functionality to easily
test and combine this with other tools such as a GUI. 

[Prequisites]
### Prequisites
  * camera compatible with opencv2 
  * python 3.x

[Trying it out]
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

[Detection Implementation]
### Detection Implementation
... In progress
