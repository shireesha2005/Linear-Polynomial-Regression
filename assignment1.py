#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

#making the data
np.random.seed(42)
X = np.linspace(-3, 3, 100).reshape(-1, 1)
y = 0.5 * X.flatten()**3 - 2 * X.flatten()**2 + X.flatten() + np.random.normal(0, 3, 100)

#spillting the data
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#creating the model
linear_model = LinearRegression()
linear_model.fit(X_train,y_train)

#making prediction and error on both train and test data
y_train_pred_linear = linear_model.predict(X_train)
y_test_pred_linear = linear_model.predict(X_test)

train_error_linear = mean_squared_error(y_train,y_train_pred_linear)
test_error_linear = mean_squared_error(y_test,y_test_pred_linear)

#printing the errors
print(f"Linear Regression training error:,{train_error_linear:.3f}")
print(f"linear regression testing error:,{test_error_linear:.3f}")

#visualization for linear regression 
X_plot = np.linspace(-3,3,200).reshape(-1,1)
y_plot_linear = linear_model.predict(X_plot)
plt.figure(figsize=(8,5))
plt.scatter(X_train,y_train,color='black',alpha=0.5,label="training data")
plt.scatter(X_test,y_test,color='red',alpha=0.5,label="testing data")
plt.plot(X_plot,y_plot_linear,color='blue',label="linear regression")
plt.title("linear regression on data")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()

#polynomial regression 
#polynomial degrees
degrees = [2, 4, 5, 10]

train_errors = []
test_errors = []
models = {}
polys = {}

for degree in degrees:
    # polynomial feature transformation
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    

    # model training
    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    # predictions
    y_train_pred = model.predict(X_train_poly)
    y_test_pred = model.predict(X_test_poly)
    

    # errors
    train_error = mean_squared_error(y_train, y_train_pred)
    test_error = mean_squared_error(y_test, y_test_pred)

    train_errors.append(train_error)
    test_errors.append(test_error)

    models[degree] = model
    polys[degree] = poly

    # printing  errors 
    print(f"Polynomial Degree {degree}")
    print(f"Training Error: {train_error:.3f}")
    print(f"Testing  Error: {test_error:.3f}")
    print("-" * 40)

X_plot = np.linspace(-3, 3, 200).reshape(-1, 1)
plt.figure(figsize=(10, 6))

for degree in degrees:
    poly = polys[degree]
    model = models[degree]

    X_plot_poly = poly.transform(X_plot)
    y_plot_pred = model.predict(X_plot_poly)

    plt.plot(X_plot, y_plot_pred, label=f"Degree {degree}")

# scatter points
plt.scatter(X_train, y_train, color='black', alpha=0.4, label="Training data")
plt.scatter(X_test, y_test, color='red', alpha=0.4, label="Testing data")

plt.title("Polynomial Regression with Increasing Degrees")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()


