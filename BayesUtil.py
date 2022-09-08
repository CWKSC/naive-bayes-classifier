import pandas as pd
from multipledispatch import dispatch


@dispatch(pd.DataFrame)
def getProbabilityDict(df: pd.DataFrame) -> dict:
    return {
        label: (df[label].value_counts() / len(df.index)).to_dict()
        for label in df.columns
    }


@dispatch(list, list)
def getProbabilityDict(colLabels: list, rowDataList: list) -> dict:
    df = pd.DataFrame(rowDataList, columns=colLabels)
    return getProbabilityDict(df)


@dispatch(pd.DataFrame)
def getConditionProbabilityDict(df: pd.DataFrame) -> dict:
    probabilityDict = getProbabilityDict(df)
    return {
        label: {
            value: getProbabilityDict(df[df[label] == value])
            for value in probabilityDict[label]
        }
        for label in df.columns
    }


@dispatch(list, list)
def getConditionProbabilityDict(colLabels: list, rowDataList: list) -> dict:
    df = pd.DataFrame(rowDataList, columns=colLabels)
    return getConditionProbabilityDict(df)
