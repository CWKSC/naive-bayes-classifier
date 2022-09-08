from typing import Any
from typing_extensions import Self
import numpy as np
import pandas as pd

import BayesUtil


class NaiveBayesClassifier:
    @staticmethod
    def fromDataFrame(df: pd.DataFrame) -> "NaiveBayesClassifier":
        object = NaiveBayesClassifier()
        object.probDict = BayesUtil.getProbabilityDict(df)
        object.condProbDict = BayesUtil.getConditionProbabilityDict(df)
        return object

    @staticmethod
    def fromListData(colLabels: list[str], rowDataList: list) -> "NaiveBayesClassifier":
        return NaiveBayesClassifier.fromDataFrame(
            pd.DataFrame(rowDataList, columns=colLabels)
        )

    def predict(
        self: Self, targetCol: str, knows: list[tuple[str, Any]]
    ) -> tuple[Any, float]:
        # Y is one of the categories
        # argmax P(y) * product of P(know|y) where y in Y, know in knows
        possibleProb = {
            value: self.probDict[targetCol][value]
            * np.prod(
                [
                    self.condProbDict[targetCol][value][knowLabel][knowValue]
                    for (knowLabel, knowValue) in knows
                ]
            )
            for value in self.probDict[targetCol].keys()
        }
        maxKey = max(possibleProb, key=possibleProb.get)
        return (maxKey, possibleProb[maxKey] / sum(possibleProb.values()))
