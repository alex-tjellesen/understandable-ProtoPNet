## What is an explanation
Given a model $ F: \mathcal{X} \to \mathcal{Y} $ and an input $ x \in \mathcal{X} $,  an explanation $ A(x; F) $ is a lower-dimensional, interpretable projection of the causal computational pathway that led to the prediction $ F(x) $. Formally,

$$
A(x; F) = P \circ T(x; F)
$$

where $ T(x; F) $ encodes the mathematical trace of computations contributing to $ F(x) $,  and $ P $ is a projection onto an interpretable subspace  $ \mathcal{Z} \subset \mathbb{R}^k $ with $ k \ll \dim(T) $.

## What makes an explanation valuable?
The non-trivial part of deriving a "good" explanation lies in the dimensionality reducing projection $P$ which gives rise to the following pitfalls:
1.	The projection $P$ may distort or hide important aspects of the underlying high-dimensional trace $T$, thereby introducing bias or misrepresentation into the resulting explanation.
2.	Even if the projection $P$ effectively reduces dimensionality, it may still fail to align with human interpretability, rendering the explanation uninformative and, ultimately, useless.

Thus a "good" explanation is an explanation that exhibits **both** (1.) high fidelity and (2.) high human interpretability, where a lack of either will not produce a useful explanation.

## How to measure the aspects of a good explanation
The following measures aims to assess the above stated criteria for a good explanation and while not being able to measure every aspect of the criteria in full arguably correlates sufficiently well with the performance on the criteria themselves.

### Measuring fidelity
We assume that the convolutional layers of the ProtoPNet aren't affected by the models goal of achieving good explanability which is a crude assumption given ProtoPNet intrinsically embedding the explanability in the model via the loss function and common backpropagation. For a given test sample $x$ and a model prediction $\hat{y}$ we attempt to reconstruct the activations of the layer preceding the prototype layer (denote this layer $\ell_{-1}$) solely based on the prototype activations. We measure the reconstruction error to the actual activations of $\ell_{-1}$ by the MSE across layer nodes and channels.