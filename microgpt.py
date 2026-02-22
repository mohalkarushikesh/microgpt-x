"""
The most atomic way to train and run inference for a GPT in pure, dependency-free Python.
This file is the complete algorithm.
Everything else is just efficiency.

@karpathy
"""

import os       # os.path.exists
import math     # math.log, math.exp
import random   # random.seed, random.choices, random.gauss, random.shuffle
random.seed(42) # Let there be order among chaos

# Let there be a Dataset `docs`: list[str] of documents (e.g. a list of names)
# Check if the dataset exists locally, otherwise download it
# if not os.path.exists('names.txt'):
#     import urllib.request
#     names_url = 'https://raw.githubusercontent.com/karpathy/makemore/master/names.txt'
#     urllib.request.urlretrieve(names_url, 'names.txt')
    
# Load names into a list and shuffle to ensure training batches aren't biased by alphabetical order
docs = [line.strip() for line in open('names.txt', encoding='utf-8') if line.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")

# Let there be a Tokenizer to translate strings to sequences of integers ("tokens") and back
# This is a character-level tokenizer: each unique character is assigned a unique ID
uchars = sorted(set(''.join(docs))) # unique characters in the dataset become token ids 0..n-1

# Define a Special Token (BOS) to signify the Start or End of a name
# We place it at the end of our character list
BOS = len(uchars) 

# The total vocabulary includes all unique characters plus the BOS token
vocab_size = len(uchars) + 1 
print(f"vocab size: {vocab_size}")
# Let there be Autograd to recursively apply the chain rule through a computation graph
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads') # Python optimization for memory usage

    def __init__(self, data, children=(), local_grads=()):
        self.data = data                # scalar value of this node calculated during forward pass
        self.grad = 0                   # derivative of the loss w.r.t. this node, calculated in backward pass
        self._children = children       # children of this node in the computation graph
        self._local_grads = local_grads # local derivative of this node w.r.t. its children

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        # The derivative of (a + b) with respect to both a and b is 1
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        # Product Rule: derivative of (a * b) w.r.t a is b, and w.r.t b is a
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other): 
        # Power Rule: d/dx(x^n) = n * x^(n-1)
        return Value(self.data**other, (self,), (other * self.data**(other-1),))
    
    def log(self): 
        # Derivative of ln(x) is 1/x
        return Value(math.log(self.data), (self,), (1/self.data,))
    
    def exp(self): 
        # Derivative of e^x is e^x
        return Value(math.exp(self.data), (self,), (math.exp(self.data),))
    
    def relu(self): 
        # Derivative is 1 if x > 0, else 0
        return Value(max(0, self.data), (self,), (float(self.data > 0),))
    
    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1

    def backward(self):
        # Build a topological sort to ensure we process nodes in the correct order
        # (A node's gradient is only finalized after all its parents are processed)
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        
        # Seed the gradient of the root node (usually the Loss) to 1.0
        self.grad = 1
        
        # Iterate backwards through the sorted list to propagate gradients to the leaves
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                # Chain Rule: dLoss/dChild = (dLoss/dV) * (dV/dChild)
                # We use += to accumulate gradients for nodes used multiple times (multivariate chain rule)
                child.grad += local_grad * v.grad

# Initialize the parameters, to store the knowledge of the model
n_layer = 1     # depth of the transformer neural network (number of layers)
n_embd = 16     # width of the network (embedding dimension)
block_size = 16 # maximum context length of the attention window (note: the longest name is 15 characters)
n_head = 4      # number of attention heads
head_dim = n_embd // n_head # derived dimension of each head

# Helper to create a matrix (list of lists) of Value objects initialized with small random weights
matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]

# Initialize the primary embedding tables and the final output head
state_dict = {
    'wte': matrix(vocab_size, n_embd), # Word Token Embeddings
    'wpe': matrix(block_size, n_embd), # Word Position Embeddings
    'lm_head': matrix(vocab_size, n_embd) # Language Model Head (logits over vocabulary)
}

# Populate the layers with Attention and MLP weights
for i in range(n_layer):
    # Projection matrices for Queries, Keys, and Values
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
    # Output projection for the combined attention heads
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
    # Feed-forward network (MLP) weights; usually 4x expansion in the hidden layer
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)

# Flatten all Value objects into a single list for the optimizer to iterate over
params = [p for mat in state_dict.values() for row in mat for p in row]
print(f"num params: {len(params)}")

# Define the model architecture: a function mapping tokens and parameters to logits
# Follow GPT-2, with minor differences: layernorm -> rmsnorm, no biases, GeLU -> ReLU

def linear(x, w):
    # Standard matrix-vector multiplication (y = Wx)
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

def softmax(logits):
    # Stabilized softmax to convert logits into a probability distribution
    max_val = max(val.data for val in logits) # Subtract max for numerical stability
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]

def rmsnorm(x):
    # Root Mean Square Normalization: scales inputs to stabilize training
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5 # 1e-5 prevents division by zero
    return [xi * scale for xi in x]

def gpt(token_id, pos_id, keys, values):
    # 1. Embedding Lookup
    tok_emb = state_dict['wte'][token_id] # Retrieve vector for current token
    pos_emb = state_dict['wpe'][pos_id]   # Retrieve vector for current position
    
    # 2. Residual Stream initialization
    # Element-wise addition of token and position embeddings
    x = [t + p for t, p in zip(tok_emb, pos_emb)] 
    x = rmsnorm(x) # Initial normalization

    for li in range(n_layer):
        # --- 1) Multi-head Attention block ---
        x_residual = x # Save for residual connection (skip connection)
        x = rmsnorm(x) # Pre-norm architecture
        
        # Project x into Query, Key, and Value spaces
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])
        
        # Cache Keys and Values for autoregressive generation (KV Cache)
        keys[li].append(k)
        values[li].append(v)
        
        x_attn = []
        # Process each head independently
        for h in range(n_head):
            hs = h * head_dim # head start index
            q_h = q[hs:hs+head_dim]
            k_h = [ki[hs:hs+head_dim] for ki in keys[li]] # All previous keys
            v_h = [vi[hs:hs+head_dim] for vi in values[li]] # All previous values
            
            # Scaled Dot-Product Attention: score = (Q @ K^T) / sqrt(d_k)
            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5 for t in range(len(k_h))]
            attn_weights = softmax(attn_logits)
            
            # Weighted sum of values based on attention scores
            head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(head_dim)]
            x_attn.extend(head_out) # Concatenate heads
            
        # Final linear projection and residual addition
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]
        
        # --- 2) MLP block (Feed-Forward) ---
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = [xi.relu() for xi in x] # Non-linearity
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)] # Residual addition

    # Final Output: Project residual stream back to vocabulary size
    logits = linear(x, state_dict['lm_head'])
    return logits
# Let there be Adam, the blessed optimizer and its buffers
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
m = [0.0] * len(params) # first moment buffer: stores the moving average of gradients (momentum)
v = [0.0] * len(params) # second moment buffer: stores the moving average of squared gradients (scaling)

# Repeat in sequence
num_steps = 1000 # number of training steps
for step in range(num_steps):

    # Take single document, tokenize it, surround it with BOS special token on both sides
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)

    # Forward the token sequence through the model, building up the computation graph all the way to the loss
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    losses = []
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax(logits)
        
        # Negative Log Likelihood (NLL) Loss: pushes the probability of the correct token toward 1.0
        loss_t = -probs[target_id].log()
        losses.append(loss_t)
    
    # final average loss over the document sequence. May yours be low.
    loss = (1 / n) * sum(losses) 

    # Backward the loss, calculating the gradients with respect to all model parameters
    # This triggers the recursive chain rule through the entire GPT graph
    loss.backward()

    # Adam optimizer update: update the model parameters based on the corresponding gradients
    lr_t = learning_rate * (1 - step / num_steps) # linear learning rate decay to fine-tune as we converge
    for i, p in enumerate(params):
        # Update first moment (momentum)
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        # Update second moment (uncentered variance)
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
        
        # Bias correction for m and v (important for the early steps of training)
        m_hat = m[i] / (1 - beta1 ** (step + 1))
        v_hat = v[i] / (1 - beta2 ** (step + 1))
        
        # Update the weight data: step in the direction of steepest descent, scaled by uncertainty
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
        
        # Zero out the gradient for the next step (crucial: gradients accumulate by default)
        p.grad = 0

    print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}", end='\r')

# Inference: may the model babble back to us
temperature = 0.5 # in (0, 1], control the "creativity" of generated text, low to high
print("\n--- inference (new, hallucinated names) ---")
for sample_idx in range(20):
    # Initialize fresh KV cache for a new sequence
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    token_id = BOS # Start with the Beginning-Of-Sentence token
    sample = []
    
    for pos_id in range(block_size):
        # Forward pass (no gradients needed here, but the graph is still built)
        logits = gpt(token_id, pos_id, keys, values)
        
        # Apply temperature: higher T = flatter distribution (more random), lower T = sharper (more deterministic)
        probs = softmax([l / temperature for l in logits])
        
        # Sample the next token based on the probability distribution
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        
        # Stop generating if the model predicts the BOS (End-Of-Sentence) token
        if token_id == BOS:
            break
        sample.append(uchars[token_id])
    
    print(f"sample {sample_idx+1:2d}: {''.join(sample)}")
