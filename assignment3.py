import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import keras
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error

####导入数据#####
train_data = pd.read_csv(r"archive\LSTM-Multivariate_pollution.csv")
test_data = pd.read_csv(r"archive\pollution_test_data1.csv")

########处理风向数据，构建训练集和测试集#########
#对风向进行独热编码（训练集）
wnd_dir_train_encoded = pd.get_dummies(train_data['wnd_dir'], prefix='wnd_dir')
#对测试集进行风向的独热编码
wnd_dir_test_encoded = pd.get_dummies(test_data['wnd_dir'], prefix='wnd_dir')
#为了保证测试集和训练集有相同的特征列，需要处理测试集中可能缺失的类别
for col in wnd_dir_train_encoded.columns:
    if col not in wnd_dir_test_encoded.columns:
        wnd_dir_test_encoded[col] = 0
wnd_dir_test_encoded = wnd_dir_test_encoded[wnd_dir_train_encoded.columns] # 确保列顺序一致
#将编码后的内容加入训练集，删除字符风向信息和时间信息
X_train = pd.concat(
    [train_data.drop(['date','wnd_dir','pollution'], axis=1), wnd_dir_train_encoded],
    axis=1)
X_test = pd.concat(
    [test_data.drop(['wnd_dir','pollution'], axis=1), wnd_dir_test_encoded],
    axis=1)
Y_train = train_data['pollution']
Y_test = test_data['pollution']

#######数据归一化#######
scaler1 = MinMaxScaler()
X_train_scaled = scaler1.fit_transform(X_train)
X_test_scaled = scaler1.transform(X_test)
scaler2 = MinMaxScaler()
Y_train_scaled = scaler2.fit_transform(Y_train.to_frame())
Y_test_scaled = scaler2.transform(Y_test.to_frame())

#####构建模型######
#将一个时间窗口内的数据结合成一个数据组，LSTM用每个数据组进行训练
def combine_data(X_in, Y_in, step):
    X, Y = [], []
    for i in range(len(X_in)-step):
        X.append(X_in[i:(i+step), :])
        Y.append(Y_in[i+step])
    return np.array(X), np.array(Y)

#根据过去12h的数据估计下一小时的污染（比24h效果略好，比72h效果好非常多）
time_step = 12
X_train_combine, Y_train_combine = combine_data(X_train_scaled,Y_train_scaled, time_step)

#构建模型：一层LSTM+Dropout+全输出层（比2层LSTM效果好）
feature = X_train_scaled.shape[1]
model = keras.Sequential([
    keras.layers.LSTM(
        units = 64,
        input_shape=[time_step, feature],
        return_sequences= False),
    keras.layers.Dropout(0.2),
    #keras.layers.LSTM(units = 32),
    keras.layers.Dense(units = 32, activation='relu'),
    #keras.layers.Dropout(0.1),
    keras.layers.Dense(units = 1)
])
#设置优化器，学习率为0.2
opt = Adam(learning_rate=0.001,clipvalue=1.0)
model.compile(optimizer=opt, loss='mse')

#学习率调度器，损失函数基本不变的时候调小学习率
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=5, min_lr=0.0000001)
#早停机制，损失函数基本不变的时候停止训练节省算力
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

#设置验证集
split_idx = int(0.8 * len(X_train_combine))
X_train, X_val = X_train_combine[:split_idx], X_train_combine[split_idx:]
Y_train, Y_val = Y_train_combine[:split_idx], Y_train_combine[split_idx:]

#######训练模型#######
history = model.fit(X_train_combine, Y_train_combine,
                    epochs=100,
                    batch_size=128,
                    validation_data=(X_val, Y_val),
                    callbacks=[lr_scheduler, early_stopping])

########在测试集上进行预测########
#X_test_combine是用于预测的特征数据组，Y_test_combine是真实值（去除前12h），
#Y_pred_scaled是估计值（去除前12h）
X_test_combine,Y_test_combine = combine_data(X_test_scaled,Y_test_scaled,time_step)
Y_pred_scaled = model.predict(X_test_combine)
#反归一化预测值，与未经归一化处理的真实值比较
Y_pred = scaler2.inverse_transform(Y_pred_scaled)
Y_true = Y_test[time_step:].reset_index(drop=True)
#计算均方误差
mse = mean_squared_error(Y_true, Y_pred)
print(f'测试集的均方误差 (MSE): {mse:.4f}')

########绘图########
#设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
#测试集上的预测值和真实值
plt.figure(figsize=(12, 6))
plt.plot(Y_true, label='真实值', color='blue')
plt.plot(Y_pred, label='预测值', color='red')
plt.title('污染情况预测结果对比')
plt.xlabel('时间（从第{}h开始预测）'.format(time_step + 1))
plt.ylabel('PM2.5浓度')
plt.legend()
plt.grid(True)
plt.show()
#模型训练损失和验证损失
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='训练损失 (loss)')
plt.plot(history.history['val_loss'], label='验证损失 (val_loss)')
plt.title('模型训练损失和验证损失')
plt.xlabel('Epoch')
plt.ylabel('损失 (MSE)')
plt.legend()
plt.grid(True)
plt.show()