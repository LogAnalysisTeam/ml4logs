import datetime


class Statistics:
    """Statistics class to store and compute the following:
    - true positive, false positive, true negative, false negative
    - precision, recall, f1-score
    """

    def __init__(self):
        self._measures = {"tp": 0, "tn": 0, "fp": 0, "fn": 0}
        self._metrics = {"precision": 0, "recall": 0, "f1": 0}

    def add_fp(self):
        """Add one false positive."""
        self._measures["fp"] += 1

    def add_tp(self):
        """Add one true positive."""
        self._measures["tp"] += 1

    def add_fn(self):
        """Add one false positive."""
        self._measures["fn"] += 1

    def add_tn(self):
        """Add one true positive."""
        self._measures["tn"] += 1

    def as_dict(self):
        self._compute_stats()
        data = {}
        data.update(self._measures)
        data.update(self._metrics)
        return data

    def as_string(self):
        """Statistics as string

        Returns:
            str: statistics
        """
        self._compute_stats()
        return ", ".join(
            [
                "Precision: {:.2f}%".format(self._metrics["precision"] * 100),
                "Recall: {:.2f}%".format(self._metrics["recall"] * 100),
                "F1-score: {:.2f}%".format(self._metrics["f1"] * 100),
                "tp: {}".format(self._measures["tp"]),
                "tn: {}".format(self._measures["tn"]),
                "fp: {}".format(self._measures["fp"]),
                "fn: {}".format(self._measures["fn"]),
            ]
        )

    def _compute_stats(self):
        self._metrics["precision"] = float("INF")
        if self._measures["tp"] + self._measures["fp"] > 0:
            self._metrics["precision"] = self._measures["tp"] / (
                    self._measures["tp"] + self._measures["fp"]
            )
        self._metrics["recall"] = float("INF")
        if self._measures["tp"] + self._measures["fn"] > 0:
            self._metrics["recall"] = self._measures["tp"] / (
                    self._measures["tp"] + self._measures["fn"]
            )
        self._metrics["f1"] = 0.
        if self._metrics["precision"] + self._metrics["recall"] > 0:
            self._metrics["f1"] = (
                    2 * self._metrics["precision"] * self._metrics["recall"] / (
                    self._metrics["precision"] + self._metrics["recall"]
            )
            )
            