from abc import abstractmethod
from copy import deepcopy
import math

class Heap:

    def __init__(self, n=None, input_list=None):
        if n is None and input_list is None:
            raise ValueError("size or input list should be not None")
        if n is not None and input_list is not None:
            raise ValueError("size or input list should be None")
        if input_list is not None:
            self.heap = input_list
            self.heap_size = len(input_list)
            self.empty_index = self.heap_size
        else:
            self.heap = [None] * n
            self.heap_size = n
            self.empty_index = 0

    def size(self):
        return self.empty_index

    def parent(self, i):
        return (i - 1) // 2

    def left(self, i):
        return 2 * i + 1

    def right(self, i):
        return 2 * i + 2

class MaxHeap(Heap):

    def __init__(self, n=None, input_list=None):
        super().__init__(n, input_list)
        if input_list is not None:
            self.build_max_heap()

    def increase_key(self, index):
        par = self.parent(index)
        while index > 0 and self.heap[par] < self.heap[index]:
            self.heap[index], self.heap[par] = self.heap[par], self.heap[index]
            par = self.parent(par)
            index = self.parent(index)

    def max_heapify(self, index):
        while True:
            l = self.left(index)
            r = self.right(index)
            new_index = index
            if l < self.empty_index and self.heap[l] > self.heap[new_index]:
                new_index = l
            if r < self.empty_index and self.heap[r] > self.heap[new_index]:
                new_index = r
            if new_index != index:
                self.heap[index], self.heap[new_index] = self.heap[new_index], self.heap[index]
                index = new_index
            else:
                break

    def change_key(self, index, key):
        self.heap[index] = key
        self.increase_key(index)
        self.max_heapify(index)

    def build_max_heap(self):
        for i in range((len(self.heap) - 1) // 2, -1, -1):
            self.max_heapify(i)

    def maximum(self):
        if self.empty_index == 0:
            raise RuntimeError("no elements exist in the heap")
        return self.heap[0]

    def extract_max(self):
        if self.empty_index == 0:
            raise RuntimeError("cannot extract max because heap is empty")
        self.empty_index -= 1
        ret = self.heap[0]
        self.heap[0] = self.heap[self.empty_index]
        self.max_heapify(0)
        return ret

    def insert(self, key):
        if self.empty_index == self.heap_size:
            raise RuntimeError("cannot insert key since the heap is full")
        self.heap[self.empty_index] = -math.inf
        self.change_key(self.empty_index, key)
        self.empty_index += 1

    @classmethod
    def heap_sort(cls, input_list):
        mah = MaxHeap(input_list=deepcopy(input_list))
        for i in range(len(input_list) - 1, -1, -1):
            input_list[i] = mah.extract_max()

class MinHeap(Heap):

    def __init__(self, n=None, input_list=None):
        super().__init__(n, input_list)
        if input_list is not None:
            self.build_min_heap()

    def decrease_key(self, index):
        par = self.parent(index)
        while index > 0 and self.heap[par] > self.heap[index]:
            self.heap[index], self.heap[par] = self.heap[par], self.heap[index]
            par = self.parent(par)
            index = self.parent(index)

    def min_heapify(self, index):
        while True:
            l = self.left(index)
            r = self.right(index)
            new_index = index
            if l < self.empty_index and self.heap[l] < self.heap[new_index]:
                new_index = l
            if r < self.empty_index and self.heap[r] < self.heap[new_index]:
                new_index = r
            if new_index != index:
                self.heap[index], self.heap[new_index] = self.heap[new_index], self.heap[index]
                index = new_index
            else:
                break

    def change_key(self, index, key):
        self.heap[index] = key
        self.decrease_key(index)
        self.min_heapify(index)

    def build_min_heap(self):
        for i in range((len(self.heap) - 1) // 2, -1, -1):
            self.min_heapify(i)

    def minimum(self):
        if self.empty_index == 0:
            raise RuntimeError("no elements exist in the heap")
        return self.heap[0]

    def extract_min(self):
        if self.empty_index == 0:
            raise RuntimeError("cannot extract max because heap is empty")
        self.empty_index -= 1
        ret = self.heap[0]
        self.heap[0] = self.heap[self.empty_index]
        self.min_heapify(0)
        return ret

    def insert(self, key):
        if self.empty_index == self.heap_size:
            raise RuntimeError("cannot insert key since the heap is full")
        self.heap[self.empty_index] = math.inf
        self.change_key(self.empty_index, key)
        self.empty_index += 1

    @classmethod
    def heap_reverse_sort(cls, input_list):
        mah = MinHeap(input_list=deepcopy(input_list))
        for i in range(len(input_list) - 1, -1, -1):
            input_list[i] = mah.extract_min()