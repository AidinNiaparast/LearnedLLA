import math

LOWER_THRESHOLD_ROOT = 0.2
LOWER_THRESHOLD_LEAF = 0.1
UPPER_THRESHOLD_ROOT = 0.5
UPPER_THRESHOLD_LEAF = 0.9
LEARNEDLLA_UPPER_THRESHOLD = 0.5 #the upper density threshold for each blackbox LLA used in LearnedLLA
MEMORY_SCALE = math.ceil(3 / LEARNEDLLA_UPPER_THRESHOLD) #the ratio of the total number of slots to the number of elements in LearnedLLA
INF = 1e12