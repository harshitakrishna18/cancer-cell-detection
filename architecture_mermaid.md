# Cancer Cell Detection - Implementation Architecture (Clean Professional)

This refined architecture follows the structural ground truth of the repository with a high-contrast palette and **Times New Roman** typography, optimized for research publications.

```mermaid
graph TD
    subgraph "Knowledge & Model Assets"
        A1["Labeled PCam Dataset"]
        A2["Trained EfficientNet-B0 Weights"]
    end

    subgraph "Analysis Pipeline"
        B["Image Pre-processing & Normalization"]
        C["Feature Extraction (EfficientNet Core)"]
        D["Binary Classifier Head (Probability Logit)"]
    end

    subgraph "Diagnostic Support Dashboard"
        E["Manual Patch Upload"]
        F["Risk Advisory Logic"]
        G["Clinical Risk Classification"]
        H["Probability Metrics & Confidence Scorer"]
        I["Diagnostic Recommendation Report"]
    end

    %% Sequential Downward Flow
    A1 --> B
    A2 --> C
    
    E --> B
    B --> C
    C --> D
    D --> F
    
    F --> G
    F --> H
    G --> I
    H --> I

    %% Styling (Academic Dark Palette)
    style A1 fill:#1a252f,color:#fff,stroke:#000,stroke-width:2.5px
    style A2 fill:#1a252f,color:#fff,stroke:#000,stroke-width:2.5px
    style B fill:#0b5345,color:#fff,stroke:#000,stroke-width:2.5px
    style C fill:#1a5276,color:#fff,stroke:#000,stroke-width:2.5px
    style D fill:#186a3b,color:#fff,stroke:#000,stroke-width:2.5px
    style E fill:#1c2833,color:#fff,stroke:#000,stroke-width:2.5px
    style F fill:#154360,color:#fff,stroke:#000,stroke-width:2.5px
    style G fill:#145a32,color:#fff,stroke:#000,stroke-width:2.5px
    style H fill:#4a235a,color:#fff,stroke:#000,stroke-width:2.5px
    style I fill:#6e2c00,color:#fff,stroke:#000,stroke-width:2.5px
```

> [!NOTE]
> All labels and group headers are now fully cleaned of parenthetical descriptions. The professional output uses bold, upscaled arrows for better visual weight in printed papers.
