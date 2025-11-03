### 경사하강법 최적화 알고리즘

| 이름           | 특징                    | 장점              | 단점          |
| ------------ | --------------------- | --------------- | ----------- |
| **SGD**      | 무작위 미니배치로 학습          | 빠름, 간단함         | 진동 심함       |
| **Momentum** | 관성 효과 추가              | Local Minima 탈출 | 튜닝 필요       |
| **AdaGrad**  | 변수별 학습률 조정            | 안정적 수렴          | 너무 느려질 수 있음 |
| **Adam**     | Momentum + RMSProp 결합 | 빠르고 안정적         | 초매개변수에 민감   |

---

### 가중치 초기화

* **Xavier 초기화** → Sigmoid, Tanh
  $$Var(W) = \frac{1}{n_{in}}$$
* **He 초기화** → ReLU
  $$Var(W) = \frac{2}{n_{in}}$$
* 초기값이 너무 작으면 기울기 소실, 너무 크면 폭주.

---

### 과적합 방지

* **Dropout**: 학습 시 일부 뉴런 비활성화 → 특정 특징에 의존 방지.
* 테스트 시에는 모든 뉴런 사용.

---

### CNN 데이터 파이프라인 (Keras 최신 방식)

* **데이터 로딩:**

  ```python
  train_ds = tf.keras.utils.image_dataset_from_directory(
      "path/train",
      image_size=(150,150),
      batch_size=32,
      labels="inferred"
  )
  ```

* **전처리:**

  ```python
  layers.Rescaling(1./255)
  ```

  → Train / Val / Test 전부 적용.

* **증강:**

  ```python
  data_augmentation = keras.Sequential([
      layers.RandomFlip("horizontal"),
      layers.RandomRotation(0.1)
  ])
  ```

  → Train에서만 적용됨 (evaluate 때는 자동 비활성).

---

### 예측

```python
img = image.load_img(path, target_size=(150,150))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
pred = model.predict(img_array)
```
