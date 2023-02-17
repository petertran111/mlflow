
import numpy as np

import mlflow
logged_model = 'runs:/db05b39e8c764acc8a78c54b29541534/model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

input = np.array([6.3,0.48,0.04,1.1,0.046,30,99,0.9928,3.24,0.36,9.6])
input = input.reshape(1, -1)

result = loaded_model.predict(input)

print(f'Predict: {result}')