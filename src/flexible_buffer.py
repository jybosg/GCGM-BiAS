import numpy as np

class FlexibleBuffer:
    def __init__(self, max_size):
        self.buffer = []
        self.max_size = max_size
        self.current_size = 0

    def append(self, new_array):
        new_array_size = new_array.size
        while self.current_size + new_array_size > self.max_size:
            oldest_array = self.buffer.pop(0)
            oldest_array_size = oldest_array.size
            self.current_size -= oldest_array_size

        self.buffer.append(new_array)
        self.current_size += new_array_size

    def get_buffer(self):
        return np.concatenate(self.buffer)
    
if __name__ == "__main__":
    # Example usage:
    buffer = FlexibleBuffer(max_size=20)

    array1 = np.array([1, 2, 3])
    array2 = np.array([4, 5, 6, 7, 8])
    array3 = np.array([9, 10, 11, 12, 13, 14, 15])

    buffer.append(array1)
    buffer.append(array2)
    buffer.append(array3)

    print(buffer.get_buffer())
