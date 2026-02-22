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
