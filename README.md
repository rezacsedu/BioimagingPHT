## Osteoarthritis quantification based on MRI images
Osteoarthritis (OA) is a degenerative joint disease and often defined as a heterogeneous group of health conditions and a phenomenal chronic disease that lead to joint symptoms and signs that are associated with defective integrity of articular cartilage, in addition to related changes in the underlying bone at the joint margins. The objective of this demo is to perform distributed deep learning for OA quantification based on MRI cohorts from the multicenter osteoarthritis study (MOST), which we hope, will help to diagnose the OA severity level of the patients. 

### Neural network architectures
The following 2 different neural network architectures are trained. 

    1. VGG19
    2. ResNet-18

### OA ground truths/labels
Due to difficulty in measurement, interpretation, and semi-quantitative grading systems in automatic knee OA detection, both quantitative gradings called Kellgren and Lawrence (KL) grading and Osteoarthritis Research Society International (OARSI) atlas are used.

    1. KL scale defines radiographic OA with a global composite score on a range of 0-4 and is correlated to incremental severity of OA being 0 signifying no presence of OA and grade 4 indicating serious OA. While 1 is doubtful narrowing, 2 is possible narrowing, and 3 is definite narrowing of joint space.
    2. The OARSI atlas provides image examples for grades for specific features of OA than assigning global scores and acts as an alternative to KL grading. This atlas grades tibiofemoral JSN and osteophytes separately for each compartment of knees as medial tibiofemoral, lateral tibiofemoral, and patellofemoral with a 0-3 scale, where 0 being nornal knee joint, 1 is mild knee joint, 2 is moderately mild, and 3 is severe knee joint.
    
### Scripts
    1. VGG_KL_test.py: OA quantification using VGG19 CNN architecture based on KL-grading
    2. ResNet_JSN_test.py: OA quantification using ResNet-18 CNN architecture based on JSN-grading
    3. utils: contains some utility functions
    4. networks: contains the nework implementations. 

### Installing dependencies
To run the examples, you need to install following Python modules and libraries.

    numpy
    tensorflow
    keras==2.2.4
    scipy
    matplotlib
    sklearn
    seaborn
    pandas
    skimage  
    PIL
    collections 

First, clone this repository (given that you have already installed the git):

    git clone https://github.com/rezacsedu/BioimagingPHT.git

Then you can install all the required libraries by issuing the following command:

 cd BioimagingPHT/
 pip3 install -r requirements.txt

### How to perform the training    
    1. python3 VGG_KL_test.py
    2. python3 ResNet_JSN_test.py
