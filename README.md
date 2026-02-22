# microgpt-x: A Path to Enlightenment

This project implements a **Transformer-based Language Model** from the absolute ground up. Every weight, every neuron, and every attention score is a standalone object that knows how to calculate its own derivative.

## 🧠 Philosophy

Most deep learning libraries (PyTorch, TensorFlow) operate on **Tensors**. MicroGPT operates on **Scalars**.

Instead of using optimized CUDA kernels, this model builds a massive computational graph of thousands of `Value` objects. When you calculate the loss, you are left with a Directed Acyclic Graph (DAG) that represents the entire history of the forward pass. Calling `.backward()` then triggers a recursive chain-rule traversal through that entire history.

---

## 🏗️ The Architecture

The model follows the GPT-2 "Pre-Norm" design with a few modern simplifications:

* **Engine:** Scalar-level Autograd.
* **Normalization:** RMSNorm (Root Mean Square Layer Normalization).
* **Activation:** ReLU (instead of GeLU for simplicity).
* **Optimizer:** Adam with bias correction and linear learning rate decay.
* **Attention:** Multi-head Causal Self-Attention with KV caching during inference.

### The Computation Flow

1. **Tokenization:** Text is broken into individual characters.
2. **Embedding:** Characters and their positions are mapped to vectors of `Value` objects.
3. **The Residual Stream:** Information flows through layers. Each layer adds its contribution to the stream rather than replacing it.
4. **Attention:** Each token "looks back" at previous tokens to gather context.
5. **MLP:** A feed-forward network allows the model to "think" about the information gathered by the attention head.

---

## 🛠️ Component Breakdown

### 1. The Autograd Engine (`Value`)

The heart of the project. It stores data and a gradient.

```math
\text{child.grad} += \text{local\_grad} \times \text{v.grad}
```

By overriding Python's magic methods (`__add__`, `__mul__`), we build the graph automatically as we perform math.

### 2. The Model (`gpt`)

A function that defines the connectivity of the Transformer. It takes a token and a position, passes it through RMSNorm, Attention, and MLP blocks, and returns the "logits" (predictions for the next character).

### 3. The Optimizer (Adam)

A "blessed" version of Stochastic Gradient Descent. It uses two buffers (momentum and variance) to ensure that weights are updated efficiently, even when the gradients are noisy or sparse.

---

## 🚀 How to Run

1. **Data:** Ensure `names.txt` is in the directory. The model will learn to hallucinate new names.
2. **Training:** The loop iterates through the dataset, calculates the Negative Log Likelihood loss, and updates parameters.
3. **Inference:** Once trained, the model samples from its own predicted probability distribution to generate text.

---

## 📊 Visualizing the Graph

```mermaid
graph LR
    A[Characters] --> B[Embeddings]
    B --> C[Transformer Layer]
    C --> D[Residual Stream]
    D --> E[Logits]
    E --> F[Cross Entropy Loss]
    F -.->|Backward Pass| G[Gradients]
    G -.->|Adam Update| B & C & E

```

# microgpt-x

```mermaid
graph TD
    subgraph Input_Processing [Input & Embedding]
        A[Token ID] --> B[WTE Lookup]
        C[Pos ID] --> D[WPE Lookup]
        B & D --> E[Add: Residual Stream]
        E --> F[RMSNorm]
    end

    subgraph Transformer_Layer [Transformer Block x Layer]
        F --> G[Pre-Norm RMSNorm]
        G --> H[Linear: Q, K, V]
        H --> I[Multi-Head Attention]
        I --> J[Linear: Projection]
        J --> K[Residual Add]
        K --> L[Pre-Norm RMSNorm]
        L --> M[Linear: FC1 + ReLU]
        M --> N[Linear: FC2]
        N --> O[Residual Add]
    end

    subgraph Output_Head [Language Model Head]
        O --> P[Final Linear: LM Head]
        P --> Q[Softmax]
        Q --> R[Cross-Entropy Loss]
    end

    subgraph Autograd [Autograd Engine: Value Class]
        R -.->|1.0| S[Backward Pass]
        S -.->|Chain Rule| T[Gradient Accumulation]
        T -.->|Update| U[Adam Optimizer]
        U -.->|New Data| B & D & H & J & M & N & P
    end

    style Autograd fill:#f9f,stroke:#333,stroke-width:2px
    style Transformer_Layer fill:#e1f5fe,stroke:#01579b

```

<!--- <img width="1384" height="2640" alt="image" src="https://github.com/user-attachments/assets/094b5af0-5955-4c63-b346-f1bfc4565273" /> --->
