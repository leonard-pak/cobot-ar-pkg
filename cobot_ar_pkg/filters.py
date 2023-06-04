class MedianFilter:
    def __init__(self, windowSize=3) -> None:
        self.buffer = [0.0] * windowSize
        middle = int(windowSize / 2)
        if windowSize % 2 == 0:
            self.getMedianElement = lambda arr: self.__medianForEvenSize(
                arr, middle)
        else:
            self.getMedianElement = lambda arr: self.__medianForOddSize(
                arr, middle)

    def __medianForEvenSize(self, arr, middle):
        return (arr[middle] + arr[middle - 1]) / 2

    def __medianForOddSize(self, arr, middle):
        return arr[middle]

    def Filtering(self, signal):
        self.buffer = self.buffer[1:]
        self.buffer.append(signal)
        return self.getMedianElement(sorted(self.buffer))
