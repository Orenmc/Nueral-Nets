#!/bin/sh
# 1 Hidden
#python ass1.py hidden:64 epoch:30 lr:0.001 norm:False l2:0.0
#python ass1.py hidden:64 epoch:30 lr:0.001 norm:True l2:0.0
#python ass1.py hidden:256 epoch:20 lr:0.001 norm:False l2:0.0
#python ass1.py hidden:256 epoch:20 lr:0.001 norm:True l2:0.0


# 2 Hidden
#python ass1.py hidden:512,64 epoch:25 lr:0.001 norm:False l2:0.0
#python ass1.py hidden:512,64 epoch:25 lr:0.001 norm:True l2:0.0
# diff learning rate!
#python ass1.py hidden:512,64 epoch:25 lr:0.01 norm:False l2:0.0
#python ass1.py hidden:512,64 epoch:25 lr:0.01 norm:True l2:0.0
#python ass1.py hidden:512,64 epoch:25 lr:0.0001 norm:False l2:0.0
#python ass1.py hidden:512,64 epoch:25 lr:0.0001 norm:True l2:0.0

# try here different wight update!
# python ass1.py hidden:512,64 epoch:25 lr:0.001 norm:False l2:0.0
# python ass1.py hidden:512,64 epoch:25 lr:0.001 norm:True l2:0.0

# 3 Hidden
#python ass1.py hidden:1024,256,32 epoch:25 lr:0.001 norm:False l2:0.0
#python ass1.py hidden:1024,256,32 epoch:25 lr:0.001 norm:True l2:0.0


############################# seconde round: ##############################

# 2 Hidden - diffirent weight initailization
#python ass1.py hidden:512,64 epoch:10 lr:0.001 norm:False l2:0.0
#python ass1.py hidden:512,64 epoch:10 lr:0.001 norm:True l2:0.0

# was bad!
# 2 Hidden - with l2
python ass1.py hidden:512,64 epoch:10 lr:0.001 norm:True l2:0.01
python ass1.py hidden:512,64 epoch:10 lr:0.001 norm:True l2:0.001

