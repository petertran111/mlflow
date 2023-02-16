
import numpy as np

import mlflow
logged_model = 'runs:/53789294254e458a8bda39d463fa49bc/model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)


input = np.array([6.3,0.48,0.04,1.1,0.046,30,99,0.9928,3.24,0.36,9.6])
input = input.reshape(1, -1)

result = loaded_model.predict(input)

print(f'Predict: {result}')