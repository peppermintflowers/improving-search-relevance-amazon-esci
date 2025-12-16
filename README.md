# Training a Teacher Model to Improve Search Relevance for the Amazon ESCI Dataset

An implementation of the teacher model based on the paper:

`Shang, Hongwei, et al. "Knowledge Distillation for Enhancing Walmart E-commerce Search Relevance Using Large Language Models." Companion Proceedings of the ACM on Web Conference 2025. 2025.`

to enhance search relevance over the Amazon ESCI dataset: 

`Reddy, Chandan K., et al. "Shopping queries dataset: A large-scale ESCI benchmark for improving product search." arXiv preprint arXiv:2206.06588 (2022).`

The teacher model accepts a concatenated text of query and product details and predicts a logit score (probability that the item is relevant to the query).

---
## Dataset

Pre-processing for data and Train-Test split is available in data_processing.ipynb.

Steps for data pre-processing we carried out:
1. Select data marked as small_version
2. Select data with locale 'us'
3. Drop records with missing fields
4. Concatenate Query, Product Title: Product Description, Product Bullet Points, Product Brand, Product Color fields
5. Split data 85% train 15% test
6. Map labels (E,S,C,I) to soft targets for relevance (1,0.5,0,0) respectively

Example of concatenated query and product details text:

`Query: !awnmower tires without rims Product Title: (Set of 2) 15x6.00-6 Husqvarna/Poulan Tire Wheel Assy .75" Bearing Product Description: No fuss. Just take off your old assembly and replace with this. No need for tubes or tire shops. No messing around with mounting these yourself. Perfect replacement for your machine. Will make your machine look and feel new. **** .75 precision ball bearings ****. Product Bullet Point: Tire size:15x6.00-6 Ply: 4 Tubeless
6x4.5 Wheel with 3/4" Precision bearings; Hub is 3" Long with .75" precision ball bearings. No grease required. Color: Husqvarna Silver
Husqvarna wheel number: 532106732 replaces 106732x645, 106732x643, 106732x417, 532141446, 532138336, 5321383-36 532125102; Husqvarna tire number 5321122073
ATW-001
Tire OD: 14.96; Tire SW: 6.3; PSI: 30; Max Load: 570 lbs. Product Brand: Antego Tire & Wheel Product Color: Husqvarna Silver`
---

## Teacher Model Fine-Tuning Details
Teacher model training code is available in train_teacher.ipynb.
* **Base model**: `meta-llama/Llama-3.2-1B`
* LoRA Adaptation: train q and v layers (trainable params: 13,631,488)
  * LoRA Rank 128
  * LoRA Dropout 0.05
* Learning rate: 1e-5
* Add score head MLP layer to compute logit score for relevance
* Train-test split % (85-15)
* Model trained over 7 epochs, checkpointed and saved upon every 20% epoch completion
---

## Evaluation

We compute Mean Square Error for two model checkpoints, after completing epoch 3 and epoch 7. A sigmoid is applied to the model's logit output to get values between 0 and 1 and they are compared against the actual soft label.

| Checkpoint Model | Train MSE<br/>computed over 1720 samples of training data | Test MSE<br/>computed over 31003 samples of testing data |
|------------------|-----------------------------------------------------------|----------------------------------------------------------|
| Epoch 3          | 0.179435                                                     | 0.174628                                                    |
| Epoch 7          | 0.151584                                                    | 0.155861                                                    |


---

## Acknowledgements and Citations

* Paper referenced: `Shang, Hongwei, et al. "Knowledge Distillation for Enhancing Walmart E-commerce Search Relevance Using Large Language Models." Companion Proceedings of the ACM on Web Conference 2025. 2025.`
* Dataset: `Reddy, Chandan K., et al. "Shopping queries dataset: A large-scale ESCI benchmark for improving product search." arXiv preprint arXiv:2206.06588 (2022).`
* Hugging Face


