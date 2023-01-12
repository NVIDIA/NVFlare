"""
Simple numpy-based code for dynamically updating histogram.
TJL 7/1/15 <tjlane@slac.stanford.edu>
"""
import sys

import numpy as np
import matplotlib.pyplot as plt

from typing import List, Tuple, Union


class DynamicHistogram(object):

    def __init__(self, bins):
        """
        Create a histogram capable of dynamically updating itself -- that is,
        adding new samples on the fly.

        Parameters
        ----------
        bins : np.ndarray
            A vector of the low/left-inclusive bin edges
        """

        self._bins = bins
        self._counts = np.zeros(bins.shape, dtype=int)

        return

    @property
    def bins(self):
        return self._bins

    @property
    def counts(self):
        return self._counts

    def add_samples(self, x):
        """
        Add a sample, 'x', to the histogram.

        Parameters
        ----------
        x : float, np.ndarray
            The sample value. Can be a vector representing multiple samples.
        """

        if type(x) is float:
            x = np.array([x])
        inds = np.digitize(x, self.bins)
        cnts = np.bincount(inds)
        self._counts[:len(cnts)] += cnts
        return

    def plot(self):
        plt.plot(self.bins, self.counts, lw=2)
        plt.xlabel('value')
        plt.ylabel('frequency')
        plt.show()


def format_value(v: float, n=3):
    fmt = "{:." + str(n) + "e}"
    return fmt.format(v)


class _Bin:

    def __init__(self):
        self.count = 0
        self.total = 0.0
        self.min = None
        self.max = None

    def record_value(self, value: float):
        self.count += 1
        self.total += value
        if self.min is None or self.min > value:
            self.min = value
        if self.max is None or self.max < value:
            self.max = value

    def get_content(self, mode='count', total=0.0):
        if self.count == 0:
            return ""
        if mode == 'count':
            return str(self.count)
        if mode == 'percent':
            return str(round(self.count/total, 2))
        if mode == 'avg':
            avg = self.total / self.count
            return format_value(avg)
        if mode == 'min':
            return format_value(self.min)
        if mode == 'max':
            return format_value(self.max)
        return "?"


class HistPool:

    def __init__(
            self,
            name: str,
            marks: Union[List[float], Tuple],
            unit: str
    ):
        self.unit = unit
        self.name = name
        self.marks = marks
        self.cat_bins = {}  # category name => list of bins

        if not marks:
            raise ValueError("marks not specified")
        if len(marks) < 2:
            raise ValueError(f"marks must have at least two numbers but got {len(marks)}")

        for i in range(1, len(marks)):
            if marks[i] <= marks[i-1]:
                raise ValueError(f"marks must contain increasing values, but got {marks}")

        # A range is defined: left <= N < right  [...)
        # [..., M1) [M1, M2) [M2, M3) [M3, ...)
        m = sys.float_info.max
        self.ranges = [(-m, marks[0])]
        self.range_names = [f"<{marks[0]}"]
        for i in range(len(marks)-1):
            self.ranges.append((marks[i], marks[i+1]))
            self.range_names.append(f"{marks[i]}-{marks[i+1]}")
        self.ranges.append((marks[-1], m))
        self.range_names.append(f">={marks[-1]}")

    def record_value(self, category: str, value: float):
        bins = self.cat_bins.get(category)
        if bins is None:
            bins = [None for _ in range(len(self.ranges))]
            self.cat_bins[category] = bins

        for i in range(len(self.ranges)):
            r = self.ranges[i]
            if r[0] <= value < r[1]:
                b = bins[i]
                if not b:
                    b = _Bin()
                    bins[i] = b
                b.record_value(value)

    def get_table(self, mode='count'):
        headers = ["category"]
        has_values = [False for _ in range(len(self.ranges))]

        # determine bins that have values in any category
        for cat_name, bins in self.cat_bins.items():
            for i in range(len(self.ranges)):
                if bins[i]:
                    has_values[i] = True

        for i in range(len(self.ranges)):
            if has_values[i]:
                headers.append(self.range_names[i])

        rows = {}
        for cat_name, bins in self.cat_bins.items():
            total_count = 0
            if mode == 'percent':
                for b in bins:
                    if b:
                        total_count += b.count

            r = [cat_name]
            for i in range(len(bins)):
                if not has_values[i]:
                    continue

                b = bins[i]
                if not b:
                    r.append("")
                else:
                    r.append(b.get_content(mode, total_count))
            rows[cat_name] = r
        return headers, rows


def new_time_pool(name: str) -> HistPool:
    marks = (0.0001, 0.0005, 0.001, 0.002, 0.004, 0.008, 0.01, 0.02, 0.04, 0.08, 0.1, 0.2, 0.4, 0.8, 1.0, 2.0)
    return HistPool(
        name=name,
        marks=marks,
        unit="second"
    )


def test_time_pool():
    p = new_time_pool("test")
    for i in range(1000):
        x = np.random.normal()
        p.record_value("random", x)

    headers, rows = p.get_table(mode='count')
    print(f"Headers {len(headers)}: {headers}")
    print(f"Rows: {rows}")
    for n, r in rows.items():
        print(f"{n}: {len(r)}")


def test_plot():
    """
    simple test, for speed!
    """

    import time

    num_bins = 20
    # num_samples = int(1e5)
    num_samples = 20

    start = time.time()
    min = -5.0
    max = 5.0

    bins = np.linspace(min, max, num_bins)
    hist = DynamicHistogram(bins)

    for i in range(num_samples):
        #x = np.random.normal()
        x = -5.000001
        print(f"sample {i}: {x}")
        # if x < min:
        #     x = min
        # if x > max:
        #     x = max

        print(f"22sample {i}: {x}")
        hist.add_samples(x)

    end = time.time()

    elapsed = end - start
    print('Added %d samples in %.3f seconds (%d bins)' % (num_samples, elapsed, num_bins))
    print('%d ns per sample (random number gen is ~300 ns)' % (1e9 * elapsed / num_samples))

    bins = hist.bins
    print(f"bins: t={type(bins)} len={len(bins)}, v={bins}")

    counts = hist.counts
    print(f"counts: t={type(counts)} len={len(counts)} v={counts}, total={np.sum(counts)}")

    percents = counts / np.sum(counts)
    print(f"percents={percents}, total={np.sum(percents)}")

    for i in range(len(bins)):
        print(f"BIN {i}: {round(bins[i], 2)}")
    hist.plot()

    return


if __name__ == '__main__':
    #test_plot()
    test_time_pool()
