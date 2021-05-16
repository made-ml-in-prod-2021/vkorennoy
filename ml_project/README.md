## Course: ML in production
# Homework-1

# Author: Viktor Korennoy (DS-21)

Original dataset is derived from Kaggle competition:
https://www.kaggle.com/loveall/cervical-cancer-risk-classification

You can download data directly from here:
'http://archive.ics.uci.edu/ml/machine-learning-databases/00383/risk_factors_cervical_cancer.csv'

Some annotations on data are provided here if needed:
https://archive.ics.uci.edu/ml/datasets/Cervical+cancer+%28Risk+Factors%29

Data is very tough and simple models do not work well, but such is life. You can check 3 different simple models here: logistic regression, random forest and simple neural network. You can assure by yourself that they never work here.

--------
How to deal with the project
------------
Install environment:
```
pip install -r requirements.txt
```
To look through the data you can find notebook in EDA folder

To train the model you can use the following command. You can choose from 3 basic models: 
- logistic regresssionran (lr)
- random forest (rf)
- simple neural network (nn)
Other models shouldn't work. But if you found some, please, let me know :)
 ```
 python train_pipeline.py model=lr
```
To create fake data you can type (you should specify number of rows):
 ```
 python tests/create_fake_data 100
```
To predict answers on existing features use following command:
```
python eval_pipeline.py
```
To run tests and check coverage:
```
pytest tests/ -v --cov
```
