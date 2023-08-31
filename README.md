# NaiveBayesClassifier
 
```
.\build_venv.ps1
.\venv\Scripts\activate
python main.py
```

```python
from NaiveBayesClassifier import NaiveBayesClassifier

# https://roger010620.medium.com/%E8%B2%9D%E6%B0%8F%E5%88%86%E9%A1%9E%E5%99%A8-naive-bayes-classifier-%E5%90%ABpython%E5%AF%A6%E4%BD%9C-66701688db02
colLabels = ["天氣", "溫度", "活動"]
rowDataList = [
    ["晴", "炎熱", "取消"],
    ["晴", "炎熱", "取消"],
    ["陰", "炎熱", "進行"],
    ["雨", "適中", "進行"],
    ["雨", "寒冷", "進行"],
    ["雨", "寒冷", "取消"],
    ["陰", "寒冷", "進行"],
    ["晴", "適中", "取消"],
    ["晴", "寒冷", "進行"],
    ["雨", "適中", "進行"],
    ["晴", "適中", "進行"],
    ["陰", "適中", "進行"],
    ["陰", "炎熱", "進行"],
    ["雨", "適中", "取消"],
]

df = pd.DataFrame(rowDataList, columns=colLabels)
naiveBayesClassifier = NaiveBayesClassifier.fromDataFrame(df)

targetCol = "活動"
knows = [("天氣", "晴"), ("溫度", "適中")]
result = naiveBayesClassifier.predict(targetCol, knows)
print(result)
```

```
('取消', 0.574468085106383)
```