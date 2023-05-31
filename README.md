# LearnedLLA
The official implementation of LearnedLLA data structure in the paper:

McCauley, S., Moseley, B., Niaparast, A. and Singh, S., 2023. Online List Labeling with Predictions. arXiv preprint arXiv:2305.10536.. 

## Datasets
We use several datasets from SNAP Large Network Dataset Collection: https://snap.stanford.edu/data/.

We use a prefix of size 2^18=262144 of each dataset as our input sequence.
The files in the Datasets folder only contain this prefix, and not the whole dataset.
All of the files are sorted based on timestamps.

## Codes
Classic_PMA.py and Adaptive_PMA.py implement Packed-Memory Array (PMA) [1] and Adaptive Packed-Memory Array (APMA) [2] data structures, respectively.
The constants used as lower and upper density thresholds are contained in the Constants.py file.
The LearnedLLA.py file contains the implementation of the LearnedLLA. The blackboxType is the type 
of blacbox LLA used, which can be 'Classic' or 'Adaptive, corresponding to PMA and APMA, respectively.
This class also has a feature isLearned, which is True when we want to use it as LearnedLLA (with predictions), 
and is False when we want to simulate a PMA or APMA using the LearnedLLA. In the latter case, all the 
elements are inserted into the first blackbox LLA, and we do not use the predictions.

## Reproducing the results
To reproduce the plots and numerical results for each dataset, run the getResults function in 
TestRealData.py with the appropriate value for datasetName. 
datasetName can be 'Gowalla Latitude', 'Gowalla LocationID', 'MOOC', 'AskUbuntu', or 'email-Eu-core'.

## References

[1] Michael A. Bender, Erik D. Demaine, and Martin Farach-Colton. Cache-oblivious B-trees. 
In Proceedings of the 41st Annual Symposium on Foundations of Computer Science (FOCS '00), 
pages 399--409, Redondo Beach, California, USA, November 2000. IEEE Computer Society.

[2] Michael A. Bender and Haodong Hu. An adaptive packed-memory array. 
ACM Trans. Database Syst., 32(4), November 2007.
