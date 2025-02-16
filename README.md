### **Cat vs. Dog Image Classification using CNN**  

This script builds a Convolutional Neural Network (CNN) using TensorFlow and Keras to classify images of cats and dogs.  

---

## **1. Loading the Dataset**  
- The dataset is loaded from CSV files (`input.csv`, `labels.csv`, `input_test.csv`, `labels_test.csv`).  
- The images are reshaped into **100x100x3** format for RGB processing.  
- The pixel values are normalized by dividing by **255.0**.  

```python
X_train = np.loadtxt('input.csv', delimiter=',')
Y_train = np.loadtxt('labels.csv', delimiter=',')

X_test = np.loadtxt('input_test.csv', delimiter=',')
Y_test = np.loadtxt('labels_test.csv', delimiter=',')

X_train = X_train.reshape(len(X_train), 100, 100, 3) / 255.0
X_test = X_test.reshape(len(X_test), 100, 100, 3) / 255.0

Y_train = Y_train.reshape(len(Y_train), 1)
Y_test = Y_test.reshape(len(Y_test), 1)
```

---

## **2. Displaying a Random Image from Training Data**  

```python
idx = random.randint(0, len(X_train))
plt.imshow(X_train[idx, :])
plt.show()
```

---

## **3. Building the CNN Model**  
- The model consists of **two convolutional layers** (with ReLU activation).  
- **MaxPooling** is applied after each convolution to reduce dimensionality.  
- The output is flattened and passed through **dense layers** with a final **sigmoid activation** for binary classification.  

```python
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(100, 100, 3)),
    MaxPooling2D((2,2)),

    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D((2,2)),

    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])
```

---

## **4. Compiling and Training the Model**  
- **Binary Crossentropy** is used as the loss function.  
- **Adam Optimizer** is applied.  
- The model is trained for **5 epochs** with a **batch size of 64**.  

```python
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=5, batch_size=64)
```

### **Training Results**  
```
Epoch 1: Accuracy = 84.75%  
Epoch 2: Accuracy = 88.00%  
Epoch 3: Accuracy = 91.60%  
Epoch 4: Accuracy = 93.95%  
Epoch 5: Accuracy = 96.10%  
```

---

## **5. Evaluating the Model on Test Data**  
```python
model.evaluate(X_test, Y_test)
```
### **Test Results**  
```
Loss: 0.9762  
Accuracy: 65.50%  
```

---

## **6. Making Predictions on a Random Test Image**  
```python
idx2 = random.randint(0, len(Y_test))
plt.imshow(X_test[idx2, :])
plt.show()

y_pred = model.predict(X_test[idx2, :].reshape(1, 100, 100, 3))
y_pred = y_pred > 0.5

if y_pred == 0:
    pred = 'dog'
else:
    pred = 'cat'

print("Our model says it is a:", pred)
```
### **Example Output:**  
```
Our model says it is a: dog
```

---

## **Conclusion**  
This CNN model effectively classifies cat and dog images, achieving **96.1% training accuracy** but lower **65.5% test accuracy**, indicating possible overfitting. It can be improved with techniques like data augmentation, dropout layers, and hyperparameter tuning.
