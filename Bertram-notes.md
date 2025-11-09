## What is an explanation
Given a model $F: \mathcal{X} \to \mathcal{Y}$ and an input $x \in \mathcal{X}$,  an explanation $A(x; F)$ is a lower-dimensional, interpretable projection of the causal computational pathway that led to the prediction $F(x)$. Formally,

$$
A(x; F) = P \circ T(x; F)
$$

where $T(x; F)$ encodes the mathematical trace of computations contributing to $F(x)$,  and $P$ is a projection onto an interpretable subspace  $\mathcal{Z} \subset \mathbb{R}^k$ with $k \ll \dim(T)$.

## What makes an explanation valuable?
The non-trivial part of deriving a "good" explanation lies in the dimensionality reducing projection $P$ which gives rise to the following pitfalls:
1.	The projection $P$ may distort or hide important aspects of the underlying high-dimensional trace $T$, thereby introducing bias or misrepresentation into the resulting explanation.
2.	Even if the projection $P$ effectively reduces dimensionality, it may still fail to align with human interpretability, rendering the explanation uninformative and, ultimately, useless.

Thus a "good" explanation is an explanation that exhibits **both** (1.) high fidelity and (2.) high human interpretability, where a lack of either will not produce a useful explanation.

## How to measure the aspects of a good explanation
The following measures aims to assess the above stated criteria for a good explanation and while not being able to measure every aspect of the criteria in full arguably correlates sufficiently well with the performance on the criteria themselves.

### Measuring fidelity
We assume that the convolutional layers of the ProtoPNet aren't affected by the models goal of achieving good explanability which is a crude assumption given ProtoPNet intrinsically embedding the explanability in the model via the loss function and common backpropagation. For a given test sample $x$ and a model prediction $\hat{y}$ we attempt to reconstruct the activations of the layer preceding the prototype layer (denote this layer $\ell_{-1}$) solely based on the prototype activations. We measure the reconstruction error to the actual activations of $\ell_{-1}$ by the MSE across layer nodes and channels.

### Measuring human interpretability
While there are multiple aspects that factor into making an explanation interpretable to humans we chose to focus on a single yet important factor: to make an explanation interpretable it must consist of just a few "concepts", which for the case of ProtoPNet means that a classification should rely on just a few prototypes and not many. For each correctly predicted test sample, we determine which prototypes contribute positively to the predicted class by computing an evidence score defined by the product of the prototype activation and it's final layer weight. We then measure what percentage of the total positive evidence comes from the top-1, top-3, and top-5 prototypes. High concentration percentages indicate that the model's decision relies on only a few key prototypes, making the explanation simpler and more interpretable to humans. We only consider correctly predicted samples to isolate the performance of the explainability layer of the model from the overall classifier.

## Experimental setup and results

**Fidelity**\
For testing fidelity we define a FCNN model and task it with learning the mapping from the prototype activations to the preceding non-prototype layer $\ell_{-1}$: 
$$
R: \mathbb{R}^{m} \to \mathbb{R}^{d \times h \times w}
$$

where $m$ is the number of prototypes, and $d \times h \times w$ are the dimensions (channels, height, width) of the activations in layer $\ell_{-1}$. The reconstruction error is then measured as:

$$\mathcal{L}_{\text{Fidelity}}(A) = \frac{1}{H \cdot W \cdot C} \sum_{h=1}^{H} \sum_{w=1}^{W} \sum_{c=1}^{C} \left( \mathbf{Z}_{h,w,c} - \hat{\mathbf{Z}}_{h,w,c} \right)^2$$

where $\mathbf{Z}$ represents the true activations of layer $\ell_{-1}$ for a given input, $\hat{\mathbf{Z}}$ represents the reconstructed activations from the prototype activations, and $H$, $W$, $C$ are the height, width, and number of channels of layer $\ell_{-1}$ respectively.

The reconstruction model $R$ is implemented with single hidden layer followed by ReLU activation. We train the model on the training set to avoid memorization, and train it for 10 epochs with a batch size of 50 and the Adam optimizer.

The model converges to a MSE of $\mathbf{0.022364}$. 

The relatively low MSE of 0.022364 indicates that the prototype activations capture a substantial portion of the information present in the preceding convolutional layer $\ell_{-1}$. This suggests that the dimensionality reduction from the full feature space to the prototype space introduces just minimal information loss. In other words, the prototype layer preserves most of the discriminative features necessary for classification, demonstrating high fidelity. However, it's important to note that this assumes the convolutional layers themselves are unaffected by the explainability objective which may not fully hold given ProtoPNet's joint optimization of both classification and interpretability through its loss function.

**Interpretability** \
For each correctly predicted test sample $x$ with predicted class $\hat{y}$, we compute the evidence vector $\mathbf{e} \in \mathbb{R}^m$ where:

$$
e_i = s_i \cdot w_{\hat{y},i}
$$

with $s_i$ being the similarity of prototype $i$ and $w_{\hat{y},i}$ the corresponding final layer weight. We measure evidence concentration as:

$$
\rho_k = \frac{\sum_{i=1}^{k} e_{(i)}}{\sum_{j: e_j > 0} e_j}
$$

where $e_{(i)}$ is the $i$-th largest evidence value and $k \in \{1, 3, 5\}$.

We find that for correctly predicted samples, on average $\mathbf{31.51\%}$ of evidence is in the top 1, $\mathbf{66.70\%}$ of evidence is in the top 3 and $\mathbf{85.21\%}$ of evidence is in the top 5.

These results demonstrate strong evidence concentration among the top prototypes, indicating that ProtoPNet produces relatively sparse and interpretable explanations. The fact that approximately two-thirds of the decision evidence comes from just 3 prototypes suggests that the model's reasoning can be effectively communicated to humans through a small set of concrete visual examples.