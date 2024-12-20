import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split

# بارگذاری داده‌ها
data = pd.read_csv('cc.csv')

# نمایش پنج ردیف اول
print(data.head())

# تبدیل مقادیر diagnosis به عددی
data['diagnosis'] = data['diagnosis'].apply(lambda x: 1 if x == 'M' else 0)

# حذف ستون‌های غیرضروری
data = data.drop(['Unnamed: 32', 'id'], axis=1)

# جداسازی متغیر هدف و ویژگی‌ها
y = data['diagnosis']
x = data.drop(['diagnosis'], axis=1)

# نرمال‌سازی داده‌ها
x = (x - np.min(x)) / (np.max(x) - np.min(x))

# تقسیم داده‌ها به مجموعه‌های آموزشی و آزمایشی
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=43)

# تعریف و آموزش مدل
model = linear_model.LogisticRegression(max_iter=10000)
model.fit(x_train, y_train)

# نمایش نتایج
print("Training accuracy:", model.score(x_train, y_train))
print("Testing accuracy:", model.score(x_test, y_test))
