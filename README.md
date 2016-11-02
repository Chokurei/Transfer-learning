# Transfer-learning
This work based on:  
https://github.com/fchollet/keras/blob/master/examples/mnist_transfer_cnn.py
=======================================
## transfer_learning.py
-------------
Using MNIST data set  
Output file: 

        ########################Time:2016-11-02-11-56########################        
        #                    File:trail.py
        ######################Transfer data ratio: 1/1########################
        Transfer learning trail, data set: MNIST digit identification
        Sample size:28 x 28
        +++++++++++++++++training original and transfered model++++++++++++++++
        Origial model:
        Data set: digit 0 ~ 4
        Number of trianing samples:30596
                  testing samples:5139
        Batch_size:128     Iteration:10
        Training time:0:00:27.787430    Every_iter:0:00:02.778743
        Validation accuracy:0.998054096128     Score:0.00651511474561

        transfered model:
        Data set: digit 5 ~ 9
        Data ratio: 1/1
        Number of trianing samples:29403
                  testing samples:4860
        Batch_size:128     Iteration:10
        Training time:0:00:09.737900    Every_iter:0:00:00.973790
        Validation accuracy:0.991769547423     Score:0.0243353945427
        +++++++++++++++++testing original and transfered model++++++++++++++++
        -----------------------group 1: testing 0::4-----------------
        Testing data amount: 30596
        Original model result:
        Confusion matrix:
        [[6899    2    0    1    1]
         [   1 7856   11    1    8]
         [  11   13 6953    6    7]
         [   2    2   16 7119    2]
         [   1    8    2    0 6813]]
        Accuracy:
        0.997341541906
        transfered model result:
        Confusion matrix:
        [[2189 2833  377 1361  143]
         [  39 2408 1772 3565   93]
         [  36  272 2049 4587   46]
         [4029    3  159 2205  745]
         [   1  678  231  291 5623]]
        Accuracy:
        0.405037078494
        -----------------------group 2: testing 5::9-----------------
        Testing data amount: 29404
        Original model result:
        Confusion matrix:
        [[ 258  133   48 5697  177]
         [1190  283  563  551 4289]
         [1528  503 2385 1376 1501]
         [ 412  908 1731 3315  459]
         [ 197   88   34  790 5849]]
        Accuracy:
        0.352838173063
        transfered model result:
        Confusion matrix:
        [[6260   23    2   21    7]
         [  14 6855    0    6    1]
         [   2    0 7268   12   11]
         [  19   12   11 6765   18]
         [  17    5   26   24 6886]]
        Accuracy:
        0.993258426966
        -----------------------group 3: testing 0::9-----------------
        Testing data amount: 70000
        Original model result:
        Confusion matrix:
        [[ 7157   135    48  5698   178]
         [ 1191  8139   574   552  4297]
         [ 1539   516  9338  1382  1508]
         [  414   910  1747 10434   461]
         [  198    96    36   790 12662]]
        Accuracy:
        0.681857142857
        transfered model result:
        Confusion matrix:
        [[ 8449  2856   379  1382   150]
         [   53  9263  1772  3571    94]
         [   38   272  9317  4599    57]
         [ 4048    15   170  8970   763]
         [   18   683   257   315 12509]]
        Accuracy:
        0.692971428571
##visualization.py
-----------------------------------
Output:  
[[https://github.com/username/repository/blob/master/img/octocat.png|alt=octocat]]
![alt text](file:///media/kaku/Work/transfer_learning/result/Acc_all_2016-11-02-19-42.html)


