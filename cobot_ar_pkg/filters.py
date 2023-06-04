
def easy_mean(f, s_k=0.5, max_k=0.9, d=1.5):
    # Creating static variable
    if not hasattr(easy_mean, "fit"):
        easy_mean.fit = f

    # Adaptive ratio
    k = s_k if (abs(f - easy_mean.fit) < d) else max_k

    # Calculation easy mean
    easy_mean.fit += (f - easy_mean.fit) * k

    return easy_mean.fit


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
        # return easy_mean(self.getMedianElement(sorted(self.buffer)))
        return self.getMedianElement(sorted(self.buffer))
