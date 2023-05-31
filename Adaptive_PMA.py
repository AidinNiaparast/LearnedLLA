import math
import Constants

class Marker:
    def __init__(self, element, count):
        self.element = element
        self.count = count

#the internal predictor in the APMA data structure    
class Predictor:
    #after the first insertion into the predictor, the following invariant is maintained:
    #the elements are in [tail, head] (both inclusive, circular), and the predictor is full iff head + 1 = tail (circular, i.e., mod self.size)
    def __init__(self, size):
        self.size = size #size is c.log(N) for some constant c, where N is the number of slots
        self.threshold = size
        self.head = 0
        self.tail = 0 
        self.array = [Marker(None,0) for i in range(self.size)]   

    def resize(self, newSize): 
        if newSize < self.size:
            print('Error: newSize < current size')
            quit()
        array = []
        i = self.tail
        array.append(self.array[i])
        while i != self.head:
            array.append(self.array[(i+1) % self.size])
            i = (i+1) % self.size
        self.tail = 0
        self.head = len(array) - 1    
        self.array = array
        while len(self.array) < newSize:
            self.array.append(Marker(None,0))
        self.size = newSize
        self.threshold = self.size    

    def insert(self, x):
        xPos = -1
        for i in range(self.size):
            if self.array[i].element == x:
                xPos = i
                break
        if xPos != -1: #x is in the predictor    
            if xPos != self.head:
                temp = self.array[xPos]
                self.array[xPos] = self.array[(xPos+1) % self.size]
                self.array[(xPos+1) % self.size] = temp
                xPos = (xPos + 1) % self.size
            self.array[xPos].count += 1    
            if self.array[xPos].count > self.threshold:
                self.array[self.tail].count -= 1
                self.array[xPos].count -= 1       
        else: #x is not in the predictor  
            if (self.head + 1) % self.size == self.tail:
                self.array[self.tail].count -= 1
            else:
                self.head = (self.head + 1) % self.size
                self.array[self.head].element = x   
                self.array[self.head].count = 1
        if self.array[self.tail].count == 0:
            self.tail = (self.tail + 1) % self.size

    #returns the insert number of x
    def I(self, x):
        for i in range(self.size):
            if self.array[i].element == x:
                return self.array[i].count
        return 0    

class Adaptive_PMA:
    #firstAssignedRank is inclusive, lastAssignedRank is not
    #base equals the index of the first slot in the bigArray that is assigned to this blackbox LLA
    #bigArray is the array that the outer LLA (i.e., LearnedLLA) uses
    def __init__(self, firstAssignedRank, lastAssignedRank, base, bigArray):
        self.capacity = lastAssignedRank - firstAssignedRank
        self.firstAssignedRank = firstAssignedRank
        self.lastAssignedRank = lastAssignedRank
        self.base = base
        self.numElem = 0
        self.numMovements = 0
        self.lowerThresholdRoot = Constants.LOWER_THRESHOLD_ROOT
        self.lowerThresholdLeaf = Constants.LOWER_THRESHOLD_LEAF
        self.upperThresholdRoot = Constants.UPPER_THRESHOLD_ROOT
        self.upperThresholdLeaf = Constants.UPPER_THRESHOLD_LEAF
        self.memoryScale = Constants.MEMORY_SCALE
        self.maxNumSlots = self.memoryScale * self.capacity #maxNumSlots is the maximum number of slots allocated to this pma
        self.initArray(bigArray)  
        self.initialSpread()  

    def __str__(self):
        retStr = ''
        for idx,x in enumerate(self.theArray[:self.effectiveLength]):
            if idx % self.smallestRangeSize == 0:
                retStr += '.'
            else:
                retStr += ' '
            retStr += str(x)
        for x in self.theArray[self.effectiveLength:]:
            retStr += '-'  
        retStr += '\n'
        if self.numElem > 0:
            retStr += "Elements: " + str(self.numElem) + " Movements: " + str(self.numMovements) + " Movements per insert: " + str(self.numMovements/self.numElem) + "\n"
        return retStr

    #overloading < operator
    #the outer LLA stores the nonempty blackbox LLAs based on this
    def __lt__(self, other):
        #this case does not matter since we only use this for nonempty blackbox LLAs
        if self.firstElement == None or other.firstElement == None: 
            return True
        return self.firstElement < other.firstElement

    def initArray(self, bigArray):
        self.theArray = bigArray[self.base: self.base + self.maxNumSlots]
        self.firstElement = None
        self.lastElement = None
        #self.firstElement = None iff self.theArray is empty; same for lastElement
        #since the elements are sorted, first and last element are smallest and largest, respectively
        for i in range(self.maxNumSlots):
            if self.theArray[i] != None:
                self.numElem += 1
                self.lastElement = self.theArray[i]
                if self.firstElement == None:
                    self.firstElement = self.theArray[i]
        self.computeCapacity()
        self.predictor = Predictor(math.floor(math.log2(self.effectiveLength)))

    #computes the number and sizes of leaves in this LLA's binary tree
    def computeCapacity(self):
        n = max(self.numElem, 2)
        self.smallestRangeSize = math.ceil(math.log2(n)) #smallestRangeSize is the size of leaves
        self.numRanges = 2 ** math.ceil(math.log2(math.ceil(n / self.smallestRangeSize))) #numRanges is the number of leaves, which should be a power of 2
        self.smallestRangeSize = math.ceil(n / self.numRanges)
        #the effectiveLength is the length of theArray that is actually used
        self.effectiveLength = self.smallestRangeSize * self.numRanges
        if self.effectiveLength > self.maxNumSlots:
            print('Error: not enough space in the adaptive PMA')
            quit()
        #Scale up the effectiveLength
        scaleFactor = min(math.floor(self.maxNumSlots/self.effectiveLength), math.floor(n / (self.upperThresholdRoot * 0.5 * self.effectiveLength)))
        self.effectiveLength *= scaleFactor
        self.smallestRangeSize *= scaleFactor   
        self.height = round(math.log2(self.numRanges))
        self.deltaUpperThreshold = (self.upperThresholdLeaf - self.upperThresholdRoot) / self.height
        self.deltaLowerThreshold = (self.lowerThresholdRoot - self.lowerThresholdLeaf) / self.height

    #this function is for spreading all the elements in theArray into theArray[0:effectiveLength]
    #we spread the elements unevenly based on the count of the marker elements
    def initialSpread(self):
        theElems = self.gatherAndRemoveElements(0,len(self.theArray))
        self.numMovements += len(theElems)
        self.recursiveUnevenPlace(theElems, 0, self.effectiveLength)

    #if element is stored gives a slot containing element
    #if element has no successor gives an empty slot in the array with no full slot after it
    #if successor of the element is stored in slot 0 then returns -1
    #if pred(element) and succ(element) are stored in successive slots, 
    #   gives the slot storing pred(element)
    #otherwise, gives an empty slot between pred(element) and succ(element)
    def getSlot(self, element):
        l = 0
        r = self.effectiveLength - 1
        while l <= r:
            m = (l + r) // 2
            originalM = m
            while m < self.effectiveLength and self.theArray[m] == None:
                m = m + 1
            #if ran out of space, desired slot cannot be past the original m
            if m == self.effectiveLength:
                r = originalM - 1
            elif self.theArray[m] < element:
                l = m + 1
            elif self.theArray[m] > element:
                r = originalM - 1
            else:
                return m
        if l == self.effectiveLength:
            return self.effectiveLength - 1        
        if self.theArray[l] != None and self.theArray[l] > element:
            if l == 0:
                return -1
            elif self.theArray[l-1] == None or self.theArray[l - 1] < element:
                return l - 1
            else:
                print("getSlot() invariant not working: returning nonempty slot " + str(l) + " which stores " + str(self.theArray[l]) + " which is larger than element " + str(element))
                quit()        
        return l

    def gatherAndRemoveElements(self, rangeLeft, rangeRight): 
        theElems = []
        for x in self.theArray[rangeLeft:rangeRight]:
            if x != None:
                theElems.append(x)
        self.theArray[rangeLeft:rangeRight] = [None] * (rangeRight - rangeLeft)
        return theElems

    #rangeLeft is inclusive; rangeRight is not
    #this function places theElems eventy throughout [rangeLeft, rangeRight)
    def recursivePlace(self, theElems, rangeLeft, rangeRight):
        if len(theElems) == 0:
            return
        medianSlot = (rangeLeft + rangeRight)//2
        if len(theElems) == 1:
            if self.theArray[medianSlot] != None:
                print("Error: overwriting previous element ", self.theArray[medianSlot], " with ", theElems[0])
                quit()
            self.theArray[medianSlot] = theElems[0]
            return
        medianElement = len(theElems)//2
        self.recursivePlace(theElems[:medianElement], rangeLeft, medianSlot)
        self.recursivePlace(theElems[medianElement:], medianSlot, rangeRight)

    #rangeLeft is inclusive; rangeRight is not
    def rebalanceUnevenly(self, rangeLeft, rangeRight):
        #gather elements
        theElems = self.gatherAndRemoveElements(rangeLeft, rangeRight)
        self.numMovements += len(theElems)
        #spread elements
        self.recursiveUnevenPlace(theElems, rangeLeft, rangeRight)

    #rangeLeft is inclusive; rangeRight is not
    #this function spreads the elements unevenly based on the count of the marker elements
    def recursiveUnevenPlace(self, theElems, rangeLeft, rangeRight):
        rangeLevel = round(math.log2((rangeRight - rangeLeft) / self.smallestRangeSize)) #rangeLevel = the distance from leaves
        if rangeLevel == 0: #this is a leaf
            #spread elements evenly in the leaf
            self.recursivePlace(theElems, rangeLeft, rangeRight) 
            return 
        if len(theElems) == 0:
            return
        leftCap = (2 ** (rangeLevel-1)) * self.smallestRangeSize #number of slots in the left child
        rightCap = (2 ** (rangeLevel-1)) * self.smallestRangeSize #number of slots in the right child
        rangeMid = (rangeLeft + rangeRight) // 2
        tau = self.getUpperThreshold(rangeLevel)
        rho = self.getLowerThreshold(rangeLevel)
        m = len(theElems)
        splitNum = math.ceil(max(leftCap * rho, m - rightCap * tau))
        leftSum = 0
        for i in range(1,splitNum + 1): #i is 1-based. We assume the elements are {x_1,...,x_m}. So x_i = theElems[i-1]
            leftSum += self.predictor.I(theElems[i-1])
        rightSum = 0
        for i in range(splitNum+1,m+1):
            rightSum += self.predictor.I(theElems[i-1])
        optValue = abs(leftSum/(leftCap - splitNum) - rightSum/(rightCap-(m-splitNum)))
        finalSplitNum = splitNum
        for i in range(splitNum + 1, math.floor(min(leftCap*tau, m - rightCap*rho)) + 1):
            leftSum += self.predictor.I(theElems[i-1])
            rightSum -= self.predictor.I(theElems[i-1])
            curValue = abs(leftSum/(leftCap - i) - rightSum/(rightCap-(m-i)))   
            #if there are a few options with the same value, choose the one closests to the middle (i.e., m/2)
            if optValue > curValue or (optValue == curValue and abs(i - m//2) < abs(finalSplitNum - m//2)):
                optValue = curValue
                finalSplitNum = i
        self.recursiveUnevenPlace(theElems[:finalSplitNum], rangeLeft, rangeMid)
        self.recursiveUnevenPlace(theElems[finalSplitNum:], rangeMid, rangeRight)

    def makeRoom(self, slot):
        #find containing range in threshold
        rangeLeft, rangeRight = self.findRangeInThreshold(slot)
        #check to see if the search was successful
        if rangeLeft == -1 and rangeRight == -1:
            self.grow()
        else:
            self.rebalanceUnevenly(rangeLeft, rangeRight)

    def grow(self):
        elements = self.gatherAndRemoveElements(0, self.effectiveLength)
        self.computeCapacity()
        self.predictor.resize(math.floor(math.log2(self.effectiveLength)))
        self.numMovements += len(elements)
        self.recursiveUnevenPlace(elements, 0, self.effectiveLength)

    #rangeLevel: distance from leaves. The rangeLevel of a leaf is 0
    def getUpperThreshold(self, rangeLevel):
        return self.upperThresholdLeaf - rangeLevel * self.deltaUpperThreshold
    
    def getLowerThreshold(self, rangeLevel):
        return self.lowerThresholdLeaf + rangeLevel * self.deltaLowerThreshold

    def rangeIsInThreshold(self, slot, rangeLevel):
        if slot == -1:
            slot = 0
        leftSlot = self.leftRangeBoundary(slot, rangeLevel)
        count = 0
        size = self.smallestRangeSize * (2 ** rangeLevel)
        for x in self.theArray[leftSlot:leftSlot + size]:
            if x != None:
                count += 1
        # +1 ensures there is room for the new element
        return (count + 1) <= size * self.getUpperThreshold(rangeLevel) and (count + 1) >= size * self.getLowerThreshold(rangeLevel)

    def leftRangeBoundary(self, slot, rangeLevel):
        if slot == -1:
            slot = 0
        size = self.smallestRangeSize * (2 ** rangeLevel)
        return math.floor(slot/size) * size

    def findRangeInThreshold(self, slot):
        if slot == -1:
            slot = 0
        rangeLevel = 0
        while rangeLevel <= self.height and self.rangeIsInThreshold(slot, rangeLevel) == False:
            rangeLevel += 1
        if rangeLevel == self.height + 1:
            return -1, -1    
        left = self.leftRangeBoundary(slot, rangeLevel)
        right = left + (2**rangeLevel) * self.smallestRangeSize
        return left, right

    def insertIntoLeaf(self, element, slot):
        if slot != -1 and self.theArray[slot] == None:
            self.theArray[slot] = element
            return
        rangeLeft = self.leftRangeBoundary(slot, 0)
        rangeRight = rangeLeft + self.smallestRangeSize
        #determine which direction to shift elements for fewer movements
        numFullSlotsAfter = 0
        #use slot + 1 here since if we shift after, our goal is to insert into slot + 1
        #as theArray[slot] contains something smaller than element
        while(slot + 1 + numFullSlotsAfter < rangeRight and self.theArray[slot + 1 + numFullSlotsAfter] != None):
            numFullSlotsAfter += 1
        numFullSlotsBefore = 0
        while(slot - numFullSlotsBefore >= rangeLeft and self.theArray[slot - numFullSlotsBefore] != None):
            numFullSlotsBefore += 1
        if numFullSlotsAfter + slot + 1 >= rangeRight and slot - numFullSlotsBefore < rangeLeft:
            print("Error: could not find location for element " + str(element) + ", slot=" + str(slot))
            print(self)
            quit()
        if numFullSlotsAfter + slot + 1 < rangeRight and (slot - numFullSlotsBefore < rangeLeft or numFullSlotsAfter < numFullSlotsBefore):
            while(numFullSlotsAfter > 0):
                self.theArray[slot + 1 + numFullSlotsAfter] = self.theArray[slot + numFullSlotsAfter]
                self.numMovements += 1
                numFullSlotsAfter -= 1
            self.theArray[slot + 1] = element
        else:
            while(numFullSlotsBefore > 0):
                self.theArray[slot - numFullSlotsBefore] = self.theArray[slot - numFullSlotsBefore + 1]
                self.numMovements += 1
                numFullSlotsBefore -= 1
            self.theArray[slot] = element

    def insert(self, element):
        slot = self.getSlot(element)
        #insert the predecessor into the predictor
        for i in reversed(range(0,slot+1)):
            if self.theArray[i] != None:
                self.predictor.insert(self.theArray[i])
                break
        #need to handle leaf separately if there is enough room
        if self.rangeIsInThreshold(slot, 0):
            self.insertIntoLeaf(element, slot)
        else:
            self.makeRoom(slot)
            slot = self.getSlot(element)
            self.insertIntoLeaf(element, slot)
        if self.lastElement == None or (self.lastElement != None and element > self.lastElement):
            self.lastElement = element
        if self.firstElement == None or (self.firstElement != None and element < self.firstElement):
            self.firstElement = element    
        self.numElem += 1    