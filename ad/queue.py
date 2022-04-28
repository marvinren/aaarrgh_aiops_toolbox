class Queue:
    def __init__(self, queue_size=2000):
        self._queue = []
        self.size = 0
        self.maxSize = queue_size

    def enqueue(self, item, ):
        if self.size < self.maxSize:
            # enqueue
            self._queue.append(item)
            self.size += 1
        else:
            # Extrusion data from queue
            del self._queue[0]
            self._queue.append(item)


    def dequeue(self):
        # dequeue
        first = self._queue[0]
        del self._queue[0]
        return first

    def get_all_data(self):
        return self._queue
