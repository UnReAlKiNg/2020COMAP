import pandas as pd  #导入pandas包并命名为pd
import numpy as np
import matplotlib.pyplot as plt
from fbprophet import Prophet  #从fbprophet中导入prophet

df1 = pd.read_csv(r'C:\Users\ADMIN\Desktop\2020COMAP\2020_Weekend2_Problems\Problem_C_Data\reformed_hair_dryer.csv')  #导入数据文件，‘’内为文件所在位置
df2 = pd.read_csv(r'C:\Users\ADMIN\Desktop\2020COMAP\2020_Weekend2_Problems\Problem_C_Data\reformed_microwave.csv')
df3 = pd.read_csv(r'C:\Users\ADMIN\Desktop\2020COMAP\2020_Weekend2_Problems\Problem_C_Data\reformed_pacifier.csv')

m1 = Prophet()
m2 = Prophet()
m3 = Prophet()

m1.add_country_holidays(country_name='US')
m2.add_country_holidays(country_name='US')
m3.add_country_holidays(country_name='US')

m1.fit(df1)
m2.fit(df2)
m3.fit(df3)

future1 = m1.make_future_dataframe(periods=365)
future2 = m2.make_future_dataframe(periods=365)
future3 = m3.make_future_dataframe(periods=365)

future1.tail()
future2.tail()
future3.tail()

forecast1 = m1.predict(future1)
forecast1[['ds','yhat','yhat_lower','yhat_upper']].tail()
forecast2 = m2.predict(future2)
forecast2[['ds','yhat','yhat_lower','yhat_upper']].tail()
forecast3 = m3.predict(future3)
forecast3[['ds','yhat','yhat_lower','yhat_upper']].tail()

m1.plot(forecast1,xlabel="year",ylabel="star_rating").show()
m1.plot_components(forecast1).show()
m2.plot(forecast2,xlabel="year",ylabel="star_rating").show()
m2.plot_components(forecast2).show()
m3.plot(forecast3,xlabel="year",ylabel="star_rating").show()
m3.plot_components(forecast3).show()
plt.show()
"""
figure1:
黑点：真实值
蓝色：预测值
浅蓝色：可能范围

figure2:
weekly中的Monday为0.3的意思就是，在trend的基础上，加0.3；Saturday为-0.3的意思就是，在trend的基础上，减0.3。因此，这条线的高低也在一定程度上反应了“销量的趋势“
"""
