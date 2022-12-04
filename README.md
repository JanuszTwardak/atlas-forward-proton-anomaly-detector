# atlas-forward-proton-anomaly-detector


### Preface

Project aims to classify anomalies in AFP events datasets. Single dataset entry contains hit coordinates of all registered charged particles that passed through the detector. Detector is made of four planes, each one collecting data individually. Plane consists of 336x80 pixels, but they are so small that passing particle often triggers more than one pixel. Normal events should mean only one proton going through detector.

Why is this important? Creating clear dataset of normal events would allow to use more machine learning and statistical techniques in CERN research. In addition to that, analyzing anomalies could also be interesting - because we do not openly define what anomaly is we might find some valuable information or relationships in this dataset too.

### About AFP experiment (mild physics included)

From CERN site:

The ATLAS Forward Proton (AFP) project promises a significant extension to the physics reach of ATLAS by tagging and measuring the momentum and emission angle of very forward protons. This enables the observation and measurement of a range of processes where one or both protons remain intact which otherwise would be difficult or impossible to study. Such processes are typically associated with elastic and diffractive scattering, where the proton radiates a virtual colorless “object,” the so-called Pomeron, which is often thought of as a non-perturbative collection of soft gluons.

<p align="center"> <img width="700" alt="3" src="https://user-images.githubusercontent.com/50775382/160122233-729fd02d-825b-43ea-ac63-8c4d37e00267.jpg"> </p>

Diffractive protons are usually scattered at very small angles (hundreds of micro radians. In order to measure them, special devices that allow for the detector movement (so-called Roman pots, RP) are commonly used. The ATLAS Forward Proton (AFP) are placed symmetrically with respect to the ATLAS IP at 204 m and 217 m. Stations located closer to the IP contain the tracking detectors, whereas the further ones are equipped with tracking and timing devices. The reconstruction resolution of tracking detectors is estimated to be of 10 and 30 μm in x and y, respectively. The precision of Time-of-Flight measurement is expected to be of about 20 ps.


<p align="center"> <img width="400" alt="3" src="https://user-images.githubusercontent.com/50775382/160123664-fd4a3ac3-0732-4032-8f72-ea9e0e7b93b4.jpg"> </p>


Readings from detectors can be very different depending which event type has occured. It can either show one, clear diffractive proton hit, a lot of different hits (shower), missing hits due to detector fault or some other strange incidents (examples show below). Our intend was to filter these events to receive dataset that contains only single proton hits. This will allow to conduct further researches on this phenomenon as well as it might find some events we would wrongly classify as anomalies, but in reality they are truly diffractive protons hits.

Examples of different, independent registered events (projected from 2D to 1D plane due to better readability):


<p align="center"> <img width="300" alt="3" src="https://user-images.githubusercontent.com/50775382/160122746-f760fff9-6372-4a52-b15c-5bd4e7ae8b2c.png"> <img width="244" alt="5" src="https://user-images.githubusercontent.com/50775382/160122789-e3e609d3-90a8-4b9c-876d-a5f5e0e67508.png"> <img width="230" alt="4" src="https://user-images.githubusercontent.com/50775382/160122801-135382cf-7f68-4818-9a88-5704c2e7540d.png"> </p>



### Problems with my other approach(convolutional neural network)

It's better to not use convolutional neural networks, as this leads to few problems:

* event hits are represented as images:
  * cannot include additional information about events,
  * matrix is sparse (from 336x80 pixels only few aren't 0):
    * harder for autoencoder to learn,
    * unoptimal in terms of computing power and speed,
* we can't predefine what anomaly event is - we want determine it without human's bias:
  * learning dataset contains some anomalies, which conflicts with the whole idea of using autoencoders to anomaly detection,
  * we can't make automatic model validation, as it is needed to check every classified event by qualified physicist who will analyze results

All of the above points make practical use of this code pretty hard thus my decision to quit further development.



## Tools used in this project
* [Poetry] Dependency management
* [hydra] Configuration files
* [pre-commit plugins] Automate code reviewing formatting
* [DVC] Data version control
* [pdoc] Automatically create an API documentation for your project

## Project structure

├── config                      
│   ├── main.yaml                   # Main configuration file
│   ├── model                       # Configurations for training model
│   │   ├── model1.yaml             # First variation of parameters to train model
│   │   └── model2.yaml             # Second variation of parameters to train model
│   └── process                     # Configurations for processing data
│   |   ├── process1.yaml           # First variation of parameters to process data
│   |   └── process2.yaml           # Second variation of parameters to process data
│   └── visualization               # Configurations for processing data
│       ├── visualization1.yaml     # First variation of parameters to process data
│       └── visualization2.yaml     # Second variation of parameters to process data
├── data            
│   ├── final                       # data after training the model
│   ├── processed                   # data after processing
│   ├── raw                         # raw data (.root files)
│   └── raw.dvc                     # DVC file of data/raw
├── docs                            # documentation for project
├── dvc.yaml                        # DVC pipeline
├── .flake8                         # configuration for flake8 - a Python formatter tool
├── .gitignore                      # ignore files that cannot commit to Git
├── Makefile                        # store useful commands to set up the environment
├── models                          # store models
├── notebooks                       # store notebooks
├── .pre-commit-config.yaml         # configurations for pre-commit
├── pyproject.toml                  # dependencies for poetry
├── README.md                       # describe your project
├── src                             # store source code
│   ├── __init__.py                 # make src a Python module 
│   ├── process.py                  # process data before training model
│   └── train_model.py              # train model
└── tests                           # store tests
    ├── __init__.py                 # make tests a Python module 
    ├── test_process.py             # test functions for process.py
    └── test_train_model.py         # test functions for train_model.py


### Set up the environment
1. Install [Poetry](https://python-poetry.org/docs/#installation)
2. Set up the environment:
```bash
make activate
make setup
```

## Run the entire pipeline
To run the entire pipeline, type:
```bash
dvc repo
```

## Version your data
Read [this article](https://towardsdatascience.com/introduction-to-dvc-data-version-control-tool-for-machine-learning-projects-7cb49c229fe0) on how to use DVC to version your data.

```bash
dvc remote add -d remote <REMOTE-URL>
```
Commit the config file:
```bash
git commit .dvc/config -m "Configure remote storage"
```
Push the data to remote storage:
```bash
dvc push 
```
Add and push all changes to Git:
```bash
git add .
git commit -m 'commit-message'
git push origin <branch>
```

# Auto-generate API documentation

To auto-generate API document for your project, run:

```bash
make docs
```
