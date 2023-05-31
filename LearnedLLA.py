import Classic_PMA
import Adaptive_PMA
import math
import Constants
from sortedcontainers import SortedList

#None represents an empty slot
#we assume the number of elements is a power of two
#ranks are 0 based, i.e., {0,1,...,n-1}

class LearnedLLA:
    #blackboxType is the type of black box LLA used. It can be 'Classic' or 'Adaptive', which correspond to PMA and APMA, respectively
    #isLearned = True if we want to use this LLA as a learned LLA and is False if we want to use it as a PMA or APMA
    #when isLearned = False, this LLA is essentially a regular (PMA or APMA) LLA, but grows to fit the current size in a
    #particularly LearnedLLA kind of way.  In particular, this simulates the LearnedLLA exactly, but
    #every item is always placed into the first blackbox LLA. this is accomplished by editing the insert function
    def __init__(self, capacity, blackboxType = 'Classic', isLearned = True):
        self.capacity = capacity #capacity is the number of elements we are going to insert. We assume this is a power of 2
        self.numElem = 0
        self.numMovements = 0
        self.nonemptyLLAs = SortedList() #nonempty blackbox LLAs are stored in this SortedList based on their first element
        self.memoryScale = Constants.MEMORY_SCALE
        self.theArray = [None] * (self.memoryScale * capacity) 
        self.LLAList = [] #list of all the blackbox LLAs
        self.blackboxType = blackboxType
        self.isLearned = isLearned
        for i in range(self.capacity):
            if self.blackboxType == 'Classic':
                lla = Classic_PMA.PMA(i, i+1, self.memoryScale * i, self.theArray)
            elif self.blackboxType == 'Adaptive':
                lla = Adaptive_PMA.Adaptive_PMA(i, i+1, self.memoryScale*i, self.theArray)
            else:
                print('Error: invalid blackboxType!')
                quit()         
            self.LLAList.append(lla)    

    def __str__(self):
        retStr = ''
        for lla in self.LLAList:
            for idx,x in enumerate(lla.theArray[:lla.effectiveLength]):
                if idx % lla.smallestRangeSize == 0:
                    retStr += '.'
                else:
                    retStr += ' '
                retStr += str(x)
            for x in lla.theArray[lla.effectiveLength:]:
                retStr += '-'  
            retStr += '|'    
        retStr += '\n'
        if self.numElem > 0:
            retStr += "Elements: " + str(self.numElem) + " Movements: " + str(self.getTotalNumOfMovemenets()) + " Movements per insert: " + str(self.getTotalNumOfMovemenets()/self.numElem) + "\n"    
        return retStr    

    def getTotalNumOfMovemenets(self):
        ret = self.numMovements
        for lla in self.LLAList:
            ret += lla.numMovements
        return ret    

    #this function gets a rank and returns the index of the blackbox LLA in the LLAList that this rank is assigned to
    def findLLAForRank(self, rank):
        l = 0
        r = len(self.LLAList)
        while l < r:
            mid = (l+r) // 2
            lla  = self.LLAList[mid]
            if lla.firstAssignedRank > rank:
                r = mid
            elif lla.lastAssignedRank <= rank:
                l = mid + 1
            else:
                return mid
        if l == r:
            print("Error: this rank is not associated to any LLA!")
            quit()           

    #returns black box LLAs containing pred and succ of the element
    #pred is the last number strictly smaller than the element and succ is the first number strictly larger than the element 
    #if the element does not have pred returns pred = 0
    #if it does not have succ returns succ = len(LLAList) - 1
    # we use the bisect_left and bisect_right functions in the SortedList library to find pred and succ of the element.
    # this is to improve the running time of the algorithm but does not have any effect on the average number of element movements
    # that we measure in our experiments  
    def findElement(self, element):
        k = len(self.LLAList)
        if self.numElem == 0:
            return 0, k-1        
        if self.nonemptyLLAs[-1].lastElement < element:
            return self.findLLAForRank(self.nonemptyLLAs[-1].firstAssignedRank), k-1
        if self.nonemptyLLAs[0].firstElement > element:
            return 0, self.findLLAForRank(self.nonemptyLLAs[0].firstAssignedRank)
        dummyList = [None] * self.memoryScale
        dummyList[0] = element
        #we only use dummyLLA to find the pred and succ of element
        if self.blackboxType == 'Classic':
            dummyLLA = Classic_PMA.PMA(0,1,0,dummyList)
        else:
            dummyLLA = Adaptive_PMA.Adaptive_PMA(0,1,0,dummyList)
        predLLAIndex = self.nonemptyLLAs.bisect_left(dummyLLA)
        if predLLAIndex == len(self.nonemptyLLAs):
            predLLA = self.nonemptyLLAs[predLLAIndex-1]
        elif self.nonemptyLLAs[predLLAIndex].firstElement > element:
            predLLA = self.nonemptyLLAs[predLLAIndex-1]
        else: #in this case, self.nonemptyLLAs[predLLAIndex].firstElement == element
            if predLLAIndex == 0:
                predLLA = self.LLAList[0]
            else:
                predLLA = self.nonemptyLLAs[predLLAIndex-1]
        succLLAIndex = self.nonemptyLLAs.bisect_right(dummyLLA)
        if succLLAIndex == len(self.nonemptyLLAs):
            if self.nonemptyLLAs[succLLAIndex - 1].lastElement > element:
                succLLA = self.nonemptyLLAs[succLLAIndex - 1]
            else:
                succLLA = self.LLAList[-1]
        elif succLLAIndex > 0 and self.nonemptyLLAs[succLLAIndex - 1].lastElement > element:
            succLLA = self.nonemptyLLAs[succLLAIndex - 1]
        elif succLLAIndex > 0:
            succLLA = self.nonemptyLLAs[succLLAIndex]   
        else: #in this case, succLLAIndex == 0
            succLLA = self.nonemptyLLAs[succLLAIndex]                         
        return self.findLLAForRank(predLLA.firstAssignedRank), self.findLLAForRank(succLLA.firstAssignedRank)
    
    #inserts an element with a given prediction
    #when we want to simulate PMA or APMA, we do not use the prediction.
    def insert(self, element, prediction = 0):
        if self.isLearned == False:
            self.insertIntoLLA(element, 0)
            self.numElem += 1
            return
        pred, succ = self.findElement(element)
        predictedLLA = self.findLLAForRank(prediction)
        if predictedLLA < pred:
            self.insertIntoLLA(element, pred)
        elif predictedLLA > succ:
            self.insertIntoLLA(element, succ)
        else:
            self.insertIntoLLA(element, predictedLLA)        
        self.numElem += 1

    #inserts into a blackbox LLA
    def insertIntoLLA(self, element, llaIndex):
        j = 0
        cond, firstLLAIndex, lastLLAIndex = self.isjthAncestorInThreshold(llaIndex, j)
        while not cond:
            j += 1
            cond, firstLLAIndex, lastLLAIndex = self.isjthAncestorInThreshold(llaIndex, j)
        if j == 0:
            self.nonemptyLLAs.discard(self.LLAList[llaIndex]) #if lla does not exist in the nonemptyLLAs, the discard function does nothing
            self.LLAList[llaIndex].insert(element)
            self.nonemptyLLAs.add(self.LLAList[llaIndex])
            return
        #else, we have to merge some LLAs     
        #gather the information from LLAs we want to merge (delete)
        for i in range(firstLLAIndex, lastLLAIndex):
            lla = self.LLAList[i]
            self.theArray[self.memoryScale * lla.firstAssignedRank : self.memoryScale * lla.lastAssignedRank] = lla.theArray
            self.numMovements += lla.numMovements
        #make a new LLA that contains all the small LLAs in the subtree of j'th ancestor of llaIndex
        if self.blackboxType == 'Classic':
            newLLA = Classic_PMA.PMA(self.LLAList[firstLLAIndex].firstAssignedRank, self.LLAList[lastLLAIndex-1].lastAssignedRank, 
                                self.LLAList[firstLLAIndex].base, self.theArray)
        else:   
            newLLA = Adaptive_PMA.Adaptive_PMA(self.LLAList[firstLLAIndex].firstAssignedRank, self.LLAList[lastLLAIndex-1].lastAssignedRank, 
                                self.LLAList[firstLLAIndex].base, self.theArray) 
        #insert the element into the new LLA
        newLLA.insert(element)   
        #delete old LLAs and add this new LLA
        self.nonemptyLLAs.discard(self.LLAList[firstLLAIndex])
        self.LLAList[firstLLAIndex] = newLLA
        self.nonemptyLLAs.add(newLLA)
        for i in reversed(range(firstLLAIndex+1, lastLLAIndex)):    
            self.nonemptyLLAs.discard(self.LLAList[i])
            self.LLAList.remove(self.LLAList[i])
        
    #this function determines if the j'th ancestor of llaIndex in the implicit binary
    #  tree is in threshold (after adding one extra element).
    # it also returns the indices of the first and last blackbox LLAs that are in 
    # the subtree of the j'th ancestor of llaIndex.   
    def isjthAncestorInThreshold(self, llaIndex, j):
        lla = self.LLAList[llaIndex]
        #firstAssignedRank is inclusive, lastAssignedRank in not
        firstRank = lla.firstAssignedRank
        lastRank = lla.lastAssignedRank
        size = lastRank - firstRank
        level = round((math.log2(size))) #distanse from leaves
        ancestorFirstRank = (firstRank >> (level + j)) << (level + j)
        ancestorLastRank = ancestorFirstRank + (size << j) 
        #firstLLAIndex is inclusive, lastLLAIndex is not
        firstLLAIndex = llaIndex
        while firstLLAIndex >= 0 and self.LLAList[firstLLAIndex].firstAssignedRank > ancestorFirstRank:
            firstLLAIndex -= 1
        lastLLAIndex = llaIndex
        while lastLLAIndex < len(self.LLAList) and self.LLAList[lastLLAIndex].lastAssignedRank <= ancestorLastRank:
            lastLLAIndex += 1
        totalNum = 0
        for i in range(firstLLAIndex, lastLLAIndex):
            totalNum += self.LLAList[i].numElem
        #test if the blackbox LLA with totalNum+1 will be within threshold
        #the following few lines are exactly how the blackbox LLAs compute their capacity
        # in the computeCapacity function
        n = max(totalNum + 1, 2)
        smallestRangeSize = math.ceil(math.log2(n))    
        numRanges = 2 ** math.ceil(math.log2(math.ceil(n / smallestRangeSize)))
        smallestRangeSize = math.ceil(n / numRanges)
        #the effectiveLength is the length of theArray that is actually used is th eblack box LLA
        effectiveLength = smallestRangeSize * numRanges
        maxNumSlots = self.memoryScale * (size << j)
        maxEffectiveLengthPossible = effectiveLength * (maxNumSlots // effectiveLength) 
        if totalNum +1 > maxEffectiveLengthPossible * Constants.LEARNEDLLA_UPPER_THRESHOLD:
            return False, firstLLAIndex, lastLLAIndex
        else:
            return True, firstLLAIndex, lastLLAIndex