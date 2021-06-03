from torchaudio.datasets import SPEECHCOMMANDS
import os

class SubsetSC(SPEECHCOMMANDS):
    def __init__(self, subset: str = None):
        super().__init__("./", download=True)

        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.join(self._path, line.strip()) for line in fileobj]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [w for w in self._walker if w not in excludes]
        elif subset == "debug":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [self._walker[w] for w in range(len(self._walker)) if self._walker[w] not in excludes and w%10 == 0]
        elif subset == "dev":
            excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
            excludes = set(excludes)
            self._walker = [self._walker[w] for w in range(len(self._walker)) if self._walker[w] not in excludes and w%1000 == 0]

def get_label_array(data):
    labels = [x[2] for x in data]
    return sorted(list(set(labels)))

def refine(data):
    return [data[i] for i in range(len(data)) if data[i][0].shape[1] == 16000 ]

