# data_science

## Circular data
### Hyperparameters
```
lr  =  1e-3
batch_size  =  None
epochs  =  30
optimizer  =  optimizers.SGD(learning_rate=lr)
loss  =  BCELoss()
metrics  = ['accuracy']
validation_data  = (x_val, y_val)
```

### 1-layer multilayer perceptron

```
Model(
	Dense((2, 1), activation=Sigmoid)
)
```

<table>
  <tr>    
    <th>Learning Curves</th>
    <th>Data</th>
    <th>Accuracy</th>
  </tr>
  <tr>
    <td rowspan="3"><img src="./assets/circular_data/circle_1-layer_output.png" width="1000" height="300" style="display: block; margin-left: auto; margin-right: auto;" /></td>
    <td>Train</td>
    <td>0.5527</td>
  </tr>
  <tr>
    <td>Validation</td>
    <td>0.54</td>
  </tr>
  <tr>
    <td>Test</td>
    <td>0.5625</td>
  </tr>
</table>

### 2-layer multilayer perceptron

```
Model(
	Dense((2, 2), activation=ReLU)
	Dense((2, 1), activation=Sigmoid)
)
```

<table>
  <tr>    
    <th>Learning Curves</th>
    <th>Data</th>
    <th>Accuracy</th>
  </tr>
  <tr>
    <td rowspan="3"><img src="./assets/circular_data/circle_2-layer_output.png" width="1000" height="300" style="display: block; margin-left: auto; margin-right: auto;" /></td>
    <td>Train</td>
    <td>0.7847</td>
  </tr>
  <tr>
    <td>Validation</td>
    <td>0.825</td>
  </tr>
  <tr>
    <td>Test</td>
    <td>0.7375</td>
  </tr>
</table>

### 3-layer multilayer perceptron

```
Model(
	Dense((2, 4), activation=ReLU)
	Dense((4, 3), activation=ReLU)
	Dense((3, 1), activation=Sigmoid)
)
```

<table>
  <tr>    
    <th>Learning Curves</th>
    <th>Data</th>
    <th>Accuracy</th>
  </tr>
  <tr>
    <td rowspan="3"><img src="./assets/circular_data/circle_3-layer_output.png" width="1000" height="300" style="display: block; margin-left: auto; margin-right: auto;" /></td>
    <td>Train</td>
    <td>0.8916</td>
  </tr>
  <tr>
    <td>Validation</td>
    <td>0.89</td>
  </tr>
  <tr>
    <td>Test</td>
    <td>0.8</td>
  </tr>
</table>

### 4-layer multilayer perceptron

```
Model(
	Dense((2, 10), activation=ReLU)
	Dense((10, 8), activation=ReLU)
	Dense((8, 5), activation=ReLU)
	Dense((5, 1), activation=Sigmoid)
)
```

<table>
  <tr>    
    <th>Learning Curves</th>
    <th>Data</th>
    <th>Accuracy</th>
  </tr>
  <tr>
    <td rowspan="3"><img src="./assets/circular_data/circle_4-layer_output.png" width="1000" height="300" style="display: block; margin-left: auto; margin-right: auto;" /></td>
    <td>Train</td>
    <td>0.9986</td>
  </tr>
  <tr>
    <td>Validation</td>
    <td>1.0</td>
  </tr>
  <tr>
    <td>Test</td>
    <td>1.0</td>
  </tr>
</table>
