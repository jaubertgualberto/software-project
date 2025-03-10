class Priority:
    """
    handle lexicographic order of keys
    """

    def __init__(self, k1, k2):
        """
        :param k1: key value
        :param k2: key value
        """
        self.k1 = k1
        self.k2 = k2

    def __lt__(self, other):
        """
        lexicographic 'lower than'
        :param other: comparable keys
        :return: lexicographic order
        """
        return self.k1 < other.k1 or (self.k1 == other.k1 and self.k2 < other.k2)

    def __le__(self, other):
        """
        lexicographic 'lower than or equal'
        :param other: comparable keys
        :return: lexicographic order
        """
        return self.k1 < other.k1 or (self.k1 == other.k1 and self.k2 <= other.k2)


class PriorityNode:
    """
    handle lexicographic order of vertices
    """

    def __init__(self, priority, vertex):
        """
        :param priority: the priority of a
        :param vertex:
        """
        self.priority = priority
        self.vertex = vertex

    def __le__(self, other):
        """
        :param other: comparable node
        :return: lexicographic order
        """
        return self.priority <= other.priority

    def __lt__(self, other):
        """
        :param other: comparable node
        :return: lexicographic order
        """
        return self.priority < other.priority


class PriorityQueue:
    def __init__(self):
        self.heap = []
        self.vertices_in_heap = []

    def empty(self):
        return len(self.heap) == 0

    def top(self):
        if not self.heap:
            return None
        return self.heap[0].vertex

    def top_key(self):
        if not self.heap:
            return Priority(float('inf'), float('inf'))
        return self.heap[0].priority

    def pop(self):
        """Pop the smallest item off the heap, maintaining the heap invariant."""
        if not self.heap:
            return None
            
        returnitem = self.heap[0]
        lastelt = self.heap.pop()
        
        # Remove the vertex of the item being returned
        if returnitem.vertex in self.vertices_in_heap:
            self.vertices_in_heap.remove(returnitem.vertex)
        
        if self.heap:
            self.heap[0] = lastelt
            self._siftup(0)
            
        return returnitem

    def insert(self, vertex, priority):
        if vertex in self.vertices_in_heap:
            # If vertex already exists, update its priority instead
            self.update(vertex, priority)
            return
            
        item = PriorityNode(priority, vertex)
        self.vertices_in_heap.append(vertex)
        self.heap.append(item)
        self._siftdown(0, len(self.heap) - 1)

    def remove(self, vertex):
        if vertex not in self.vertices_in_heap:
            return
            
        # Find and remove the vertex from both structures
        self.vertices_in_heap.remove(vertex)
        for index, priority_node in enumerate(self.heap):
            if priority_node.vertex == vertex:
                self.heap[index] = self.heap[-1]
                self.heap.pop()
                if index < len(self.heap):
                    self._siftup(index)
                    self._siftdown(0, index)
                break

    def update(self, vertex, priority):
        if vertex not in self.vertices_in_heap:
            self.insert(vertex, priority)
            return
            
        for index, priority_node in enumerate(self.heap):
            if priority_node.vertex == vertex:
                self.heap[index].priority = priority
                self._siftup(index)
                self._siftdown(0, index)
                break

    # !!!THIS FUNCTION WAS COPIED AND MODIFIED!!! Source: Lib/heapq.py
    def build_heap(self):
        """Transform list into a heap, in-place, in O(len(x)) time."""
        n = len(self.heap)
        # Transform bottom-up.  The largest index there's any point to looking at
        # is the largest with a child index in-range, so must have 2*i + 1 < n,
        # or i < (n-1)/2.  If n is even = 2*j, this is (2*j-1)/2 = j-1/2 so
        # j-1 is the largest, which is n//2 - 1.  If n is odd = 2*j+1, this is
        # (2*j+1-1)/2 = j so j-1 is the largest, and that's again n//2-1.
        for i in reversed(range(n // 2)):
            self._siftup(i)

    # !!!THIS FUNCTION WAS COPIED AND MODIFIED!!! Source: Lib/heapq.py
    # 'heap' is a heap at all indices >= startpos, except possibly for pos.  pos
    # is the index of a leaf with a possibly out-of-order value.  Restore the
    # heap invariant.
    def _siftdown(self, startpos, pos):
        newitem = self.heap[pos]
        # Follow the path to the root, moving parents down until finding a place
        # newitem fits.
        while pos > startpos:
            parentpos = (pos - 1) >> 1
            parent = self.heap[parentpos]
            if newitem < parent:
                self.heap[pos] = parent
                pos = parentpos
                continue
            break
        self.heap[pos] = newitem

    def _siftup(self, pos):
        endpos = len(self.heap)
        startpos = pos
        newitem = self.heap[pos]
        # Bubble up the smaller child until hitting a leaf.
        childpos = 2 * pos + 1  # leftmost child position
        while childpos < endpos:
            # Set childpos to index of smaller child.
            rightpos = childpos + 1
            if rightpos < endpos and not self.heap[childpos] < self.heap[rightpos]:
                childpos = rightpos
            # Move the smaller child up.
            self.heap[pos] = self.heap[childpos]
            pos = childpos
            childpos = 2 * pos + 1
        # The leaf at pos is empty now.  Put newitem there, and bubble it up
        # to its final resting place (by sifting its parents down).
        self.heap[pos] = newitem
        self._siftdown(startpos, pos)
