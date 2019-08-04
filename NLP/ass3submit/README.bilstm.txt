###################
NAME:	Oren Cohen
ID:	305164295
USER:	cohenorx
###################

I did not use any parametes for running, but be aware for createing top models of POS and NER you need to run (only) the bilstmTag.py Twice 
one for NER and one for POS

example: bilstmTrainer.py d pos/train pos_d_model
and do this to all 8 options [Pos NER] * [a b c d]

and in the and choose the best model!

and run:
bilstmTag.py c ner_c_model ner/test
bilstmTag.py d pos_d_model pos/test


make sure that the train/dev/test sets are in the workspace:

pos/
pos/train
pos/sev
pos/test

ner/
ner/train
ner/sev
ner/test
