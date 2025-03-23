import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train_data = pd.read_excel('Data4Regression.xlsx',sheet_name='Training Data')
test_data = pd.read_excel('Data4Regression.xlsx',sheet_name='Test Data')

x_train = train_data['x'].values
y_train = train_data['y_complex'].values
x_test = test_data['x_new'].values
y_test = test_data['y_new_complex'].values

##########结果输出与绘图##########
def figure_plot(method,theta_0,theta_1,title1,title2):
    global x_train,y_train,x_test,y_test
    # 计算拟合数据
    train_re = theta_1 * x_train + theta_0
    test_re = theta_1 * x_test + theta_0
    # 计算误差
    train_error = np.mean((y_train - train_re) ** 2)
    test_error = np.mean((y_test - test_re) ** 2)
    print(f"{method}训练误差: {train_error:.8f}")
    print(f"{method}测试误差: {test_error:.8f}")

    plt.figure(figsize=(12, 12))
    #训练数据
    plt.subplot(2, 1, 1)
    plt.scatter(x_train, y_train, color='blue')
    plt.plot(x_train, train_re, color='red', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title1)
    #测试数据
    plt.subplot(2, 1, 2)
    plt.scatter(x_test, y_test, color='green')
    plt.plot(x_test, test_re, color='red', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title2)

    plt.tight_layout()
    plt.show()

###############最小二乘OLS###############
X = np.vstack([np.ones(len(x_train)), x_train]).T
theta_ols = np.linalg.inv(X.T @ X) @ X.T @ y_train
theta_ols_0,theta_ols_1 = theta_ols

figure_plot("OLS",theta_ols_0,theta_ols_1,"OLS_Training","OLS_Test")

##########梯度下降GD###########
def gradient_descent(x, y, lr=0.01, epochs=100):

    theta_0 = 0
    theta_1 = 0
    n = len(x)

    for _ in range(epochs):
        y_pred = theta_1 * x + theta_0

        grad_theta1 = (2 / n) * np.sum((y_pred-y) * x)
        grad_theta0 = (2 / n) * np.sum(y_pred-y)

        theta_1 = theta_1 - lr * grad_theta1
        theta_0 = theta_0 - lr * grad_theta0

    return theta_0, theta_1

GD_theta_0,GD_theta_1 = gradient_descent(x_train, y_train)

figure_plot("GD",GD_theta_0,GD_theta_1,"GD_Training","GD_Test")

#########牛顿法NT############
def newton_method(x, y, max_iter=100, tol=1e-6):
    n = len(x)
    theta1 = 0
    theta0 = 0

    for _ in range(max_iter):
        y_pred = theta1 * x + theta0

        grad_theta1 = (2 / n) * np.sum((y_pred-y) * x)
        grad_theta0 = (2 / n) * np.sum(y_pred-y)

        hessian11 = (2 / n) * np.sum(x ** 2)
        hessian10 = (2 / n) * np.sum(x)
        hessian00 = 2
        hessian = np.array([[hessian00,hessian10],
                           [hessian10,hessian11]])

        delta = np.linalg.inv(hessian) @ np.array([grad_theta0, grad_theta1]).T
        theta0 = theta0 - delta[0]
        theta1 = theta1 - delta[1]

        # 检查收敛
        if np.linalg.norm(delta) < tol:
            break

    return theta0,theta1

NT_theta_0,NT_theta_1 = newton_method(x_train, y_train)
figure_plot("NT",NT_theta_0,NT_theta_1,"NT_Training","NT_Test")

################高次多项式拟合#####################
def polynomial(n):
    global x_train,x_test,y_test,y_train
    degree = n
    X = np.zeros((len(x_train),degree+1))
    for i in range(degree+1):
        X[:,i] = x_train ** i
    theta = np.linalg.inv(X.T @ X) @X.T @ y_train

    train_re = np.zeros(len(x_train))
    test_re = np.zeros(len(x_test))
    for j in range(degree+1):
        train_re = train_re +theta[j] * (x_train ** j)
        test_re = test_re +theta[j] * (x_test ** j)

    # 计算误差
    train_error = np.mean((y_train - train_re) ** 2)
    test_error = np.mean((y_test - test_re) ** 2)
    print(f"{degree}次训练误差: {train_error:.8f}")
    print(f"{degree}次测试误差: {test_error:.8f}")

    plt.figure(figsize=(12, 12))
    # 训练数据
    plt.subplot(2, 1, 1)
    plt.scatter(x_train, y_train, color='blue')
    plt.plot(x_train, train_re, color='red', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f"{degree}_training data")
    # 测试数据
    plt.subplot(2, 1, 2)
    plt.scatter(x_test, y_test, color='green')
    plt.plot(x_test, test_re, color='red', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f"{degree}_training data")

    plt.tight_layout()
    plt.show()

for i in range(5,15):
    polynomial(i)