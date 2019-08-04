###########################	tagger 1 ################################

﻿For the first part I created some python files (and classes):
ReadFiles.py – for saving time, I read the files and save them compressed (with pickle)
creates 10 files: word_set, tag_set, train, dev, test x2 (NER and POS)
I saved the pickle files and submit them as well.

tagger1.py – init the dataset loaders (with NER and POS sets) create nn model and run it.

NNClass.py – class of neural network

NNTrainer.py – class, create object that can: train, dev and test (and create all the out files and graphs)

utils.py – all the functionality that I needed (globals and “global function”


for run the code simply ReadFiles.py code- and only then run tagger1.py

(for readfiles running- make sure the files are in data file in the workspace :
data – {ner, pos} – {train, dev, test}
something like this:
data→ner→train
data→ner→test
data→pos→train
...

make sure that everything is lowercase!


######################### tagger 2 ######################################

run the tagger2.py code (all the file should be in the workspace.. *.p after the first ReadFiles.py)
