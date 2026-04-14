# Arquitectura para Entrenamiento de GNN NATIVO en MillenniumDB
## Entrenamiento Completo Dentro del DBMS con Persistencia en Disco

**Fecha**: 18 de Octubre, 2025
**Versión**: 1.0
**Objetivo**: Diseñar un sistema de entrenamiento de GNN completamente integrado en MillenniumDB

---

## 📋 Tabla de Contenidos

1. [Visión y Diferenciación](#1-visión-y-diferenciación)
2. [Arquitectura de Alto Nivel](#2-arquitectura-de-alto-nivel)
3. [Componentes del Motor de GNN](#3-componentes-del-motor-de-gnn)
4. [Sistema de Persistencia de Modelos](#4-sistema-de-persistencia-de-modelos)
5. [Implementación Detallada](#5-implementación-detallada)
6. [Ejemplos de Uso](#6-ejemplos-de-uso)
7. [Ventajas Competitivas](#7-ventajas-competitivas)
8. [Roadmap de Implementación](#8-roadmap-de-implementación)

---

## 1. Visión y Diferenciación

### 1.1 Concepto Revolucionario

**Idea Central**:
```
GNN como primitiva de primera clase en el DBMS
= Entrenar modelos directamente en el grafo persistente
= Modelo entrenado persiste en disco junto con los datos
```

**Filosofía**:
```
Neo4j GDS: Proyección (RAM) → Algoritmo (CPU) → Resultado volátil
MillenniumDB: Proyección (Disco) → GNN (CPU/GPU) → Modelo persistente
```

---

### 1.2 Ventajas Únicas de Entrenar en el DBMS

| Aspecto | Entrenamiento Externo (PyTorch) | Entrenamiento Nativo (MillenniumDB) |
|---------|--------------------------------|-------------------------------------|
| **Persistencia del modelo** | Manual (guardar .pth) | Automática (catálogo del DBMS) |
| **Versionado** | Manual (MLflow, etc.) | Integrado (snapshots de proyección) |
| **Reproducibilidad** | Difícil (deps, semillas) | Garantizada (grafo + modelo versionados) |
| **Inferencia** | Cargar modelo + datos | Inmediata (modelo ya en DBMS) |
| **Escalabilidad** | Limitada por RAM/GPU | Limitada por disco (mucho mayor) |
| **Multi-tenancy** | Complejo | Natural (múltiples proyecciones) |
| **Consultas híbridas** | Imposible | Posible (SQL + ML en una query) |

**Caso de uso killer**:
```gql
-- Query que combina búsqueda estructurada + predicción ML
USE "social_network"
MATCH (u:User {country: "Chile"})
WHERE u.age > 25
WITH u
CALL mdb.gnn.predict("user_classifier_v3", u) YIELD prediction, probability
WHERE probability > 0.8
RETURN u.name, prediction, probability
ORDER BY probability DESC
LIMIT 10
```

---

### 1.3 Comparación con Neo4j GDS

**Neo4j GDS**:
- ✅ GraphSAGE nativo (Java)
- ✅ HashGNN nativo (Java)
- ❌ Solo 2 arquitecturas fijas
- ❌ No permite arquitecturas custom
- ❌ Modelos volátiles (se pierden al reiniciar)

**MillenniumDB (Propuesto)**:
- ✅ GraphSAGE nativo (C++)
- ✅ HashGNN nativo (C++)
- ✅ GCN, GAT, GIN nativos (C++)
- ✅ **Arquitecturas custom definibles por usuario** (via DSL)
- ✅ **Modelos persistentes en disco**
- ✅ **Snapshots de modelo + datos**

---

## 2. Arquitectura de Alto Nivel

### 2.1 Stack Tecnológico

```
┌─────────────────────────────────────────────────────────────────┐
│                    GQL Query Interface                          │
│  CALL mdb.gnn.train(...) / CALL mdb.gnn.predict(...)          │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                  GNN Execution Engine (C++)                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Model Registry & Catalog                                 │  │
│  │ - model_catalog.json: metadata de modelos                │  │
│  │ - model_weights/: binarios de pesos                      │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Training Pipeline                                         │  │
│  │ - Forward pass                                            │  │
│  │ - Backward pass (autodiff)                                │  │
│  │ - Optimizer (Adam, SGD)                                   │  │
│  │ - Checkpointing                                           │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Inference Pipeline                                        │  │
│  │ - Load model from disk                                    │  │
│  │ - Predict en batch/stream                                 │  │
│  │ - Cache de embeddings                                     │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│          Tensor Computation Backend (CPU/GPU)                   │
│                                                                 │
│  Opción A: LibTorch (PyTorch C++ API)                          │
│    - Ventaja: Ecosistema completo, GPU, autodiff              │
│    - Desventaja: Dependencia grande (~500MB)                  │
│                                                                 │
│  Opción B: ONNX Runtime                                        │
│    - Ventaja: Ligero, inferencia rápida                       │
│    - Desventaja: Solo inferencia (no entrenamiento)           │
│                                                                 │
│  Opción C: Custom (Eigen + autodiff manual)                    │
│    - Ventaja: Control total, ligero                            │
│    - Desventaja: Mucho trabajo, solo CPU                      │
│                                                                 │
│  **RECOMENDACIÓN: Opción A (LibTorch)**                        │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────────┐
│                 Graph Storage Layer                             │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ Projection B+Trees                                        │  │
│  │ - node_id, from_to_edge, to_from_edge                    │  │
│  │ - node_key_value (propiedades originales)                │  │
│  │ - node_computed (propiedades calculadas: embeddings)     │  │
│  └──────────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ VectorStore                                               │  │
│  │ - Embeddings intermedios (cache)                          │  │
│  │ - Embeddings finales (resultados de inferencia)           │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

### 2.2 Flujo de Datos: Entrenamiento

```
┌────────────────────────────────────────────────────────────────┐
│ 1. PREPARACIÓN                                                 │
│    GQL> USE "my_graph"                                         │
│    GQL> CALL mdb.gnn.define("my_classifier", {                │
│           architecture: "GCN",                                 │
│           layers: [64, 32],                                    │
│           task: "node_classification"                          │
│        })                                                      │
└────────────────────────┬───────────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────────────┐
│ 2. CONFIGURACIÓN DE DATOS                                      │
│    GQL> CALL mdb.gnn.configure_data("my_classifier", {        │
│           node_features: ["age", "pagerank"],                  │
│           node_labels: "community",                            │
│           train_mask: "is_train",                              │
│           val_mask: "is_val",                                  │
│           test_mask: "is_test"                                 │
│        })                                                      │
└────────────────────────┬───────────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────────────┐
│ 3. ENTRENAMIENTO                                               │
│    GQL> CALL mdb.gnn.train("my_classifier", {                 │
│           epochs: 200,                                         │
│           learning_rate: 0.01,                                 │
│           batch_size: 1024,                                    │
│           early_stopping: true                                 │
│        }) YIELD epoch, train_loss, val_loss, val_acc          │
│                                                                │
│    Internamente:                                               │
│    ┌─────────────────────────────────────────────────┐        │
│    │ Para cada epoch:                                 │        │
│    │   1. Cargar batch de nodos (stream de B+Tree)   │        │
│    │   2. Construir subgrafo (vecindad k-hop)        │        │
│    │   3. Forward pass (LibTorch)                     │        │
│    │   4. Compute loss                                │        │
│    │   5. Backward pass (autodiff)                    │        │
│    │   6. Update weights (optimizer)                  │        │
│    │   7. Checkpoint cada N epochs → disco            │        │
│    └─────────────────────────────────────────────────┘        │
└────────────────────────┬───────────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────────────┐
│ 4. PERSISTENCIA AUTOMÁTICA                                     │
│    Modelo guardado en:                                         │
│    data/dbs/gql/mi_db/projections/my_graph/models/            │
│    ├── my_classifier_v1.pt        # Pesos del modelo          │
│    ├── my_classifier_v1.json      # Arquitectura + hiperparams│
│    └── model_catalog.json         # Registro de todos modelos │
└────────────────────────────────────────────────────────────────┘
```

---

### 2.3 Flujo de Datos: Inferencia

```
┌────────────────────────────────────────────────────────────────┐
│ 1. PREDICCIÓN EN BATCH                                         │
│    GQL> USE "my_graph"                                         │
│    GQL> CALL mdb.gnn.predict("my_classifier")                 │
│         YIELD nodeId, prediction, probability                  │
│         MUTATE PROPERTY predicted_class = prediction           │
│                                                                │
│    Internamente:                                               │
│    ┌─────────────────────────────────────────────────┐        │
│    │ 1. Cargar modelo desde disco (cache en RAM)     │        │
│    │ 2. Stream de nodos desde proyección             │        │
│    │ 3. Batch inference (1000 nodos a la vez)        │        │
│    │ 4. Escribir resultados a B+Tree                 │        │
│    └─────────────────────────────────────────────────┘        │
└────────────────────────┬───────────────────────────────────────┘
                         │
                         ▼
┌────────────────────────────────────────────────────────────────┐
│ 2. PREDICCIÓN EN QUERY (Inline)                                │
│    GQL> MATCH (u:User {country: "Chile"})                     │
│         WITH u                                                 │
│         CALL mdb.gnn.predict("my_classifier", u)              │
│         YIELD prediction, probability                          │
│         WHERE probability > 0.9                                │
│         RETURN u.name, prediction                              │
│                                                                │
│    Internamente:                                               │
│    ┌─────────────────────────────────────────────────┐        │
│    │ 1. Para cada nodo u en MATCH:                   │        │
│    │    a) Extraer features de u                      │        │
│    │    b) Construir vecindad (1-2 hops)             │        │
│    │    c) Forward pass (solo lectura)               │        │
│    │    d) Retornar prediction como binding          │        │
│    └─────────────────────────────────────────────────┘        │
└────────────────────────────────────────────────────────────────┘
```

---

## 3. Componentes del Motor de GNN

### 3.1 Model Registry & Catalog

**Archivo**: `src/graph_models/gql/ml/model_catalog.h`

```cpp
// src/graph_models/gql/ml/model_catalog.h
#pragma once

#include <string>
#include <vector>
#include <memory>
#include <nlohmann/json.hpp>

namespace GQL {
namespace ML {

class ModelCatalog {
public:
    struct ModelMetadata {
        std::string model_id;              // "my_classifier_v1"
        std::string model_name;            // "my_classifier"
        int version;                       // 1
        std::string architecture;          // "GCN", "GAT", "GraphSAGE"
        std::string task;                  // "node_classification", "link_prediction"

        // Arquitectura
        std::vector<int> hidden_dims;      // [64, 32]
        int input_dim;                     // 66 (calculado)
        int output_dim;                    // 10 (número de clases)

        // Hiperparámetros de entrenamiento
        nlohmann::json hyperparameters;    // {epochs: 200, lr: 0.01, ...}

        // Features usadas
        std::vector<std::string> node_features;
        std::string label_property;

        // Métricas de entrenamiento
        float best_val_loss;
        float best_val_accuracy;
        int best_epoch;

        // Metadata
        std::string created_at;            // Timestamp ISO8601
        std::string projection_name;       // Proyección usada para entrenar
        std::string projection_snapshot;   // Hash del estado de la proyección

        // Paths en disco
        std::string weights_path;          // "models/my_classifier_v1.pt"
        std::string config_path;           // "models/my_classifier_v1.json"
    };

    ModelCatalog(const std::string& projection_dir);

    // Registrar nuevo modelo
    void register_model(const ModelMetadata& metadata);

    // Listar modelos
    std::vector<ModelMetadata> list_models() const;

    // Obtener metadata de un modelo
    ModelMetadata get_model(const std::string& model_name, int version = -1) const;

    // Verificar si modelo existe
    bool model_exists(const std::string& model_name, int version = -1) const;

    // Obtener última versión de un modelo
    int get_latest_version(const std::string& model_name) const;

    // Borrar modelo
    void delete_model(const std::string& model_name, int version);

private:
    std::string catalog_path;              // "models/model_catalog.json"
    std::vector<ModelMetadata> models;

    void load_from_disk();
    void save_to_disk();
};

} // namespace ML
} // namespace GQL
```

**Implementación**: `src/graph_models/gql/ml/model_catalog.cc`

```cpp
// src/graph_models/gql/ml/model_catalog.cc
#include "model_catalog.h"
#include <fstream>
#include <algorithm>

using json = nlohmann::json;

namespace GQL {
namespace ML {

ModelCatalog::ModelCatalog(const std::string& projection_dir)
    : catalog_path(projection_dir + "/models/model_catalog.json")
{
    // Crear directorio models/ si no existe
    std::filesystem::create_directories(projection_dir + "/models");

    load_from_disk();
}

void ModelCatalog::register_model(const ModelMetadata& metadata) {
    // Verificar si ya existe mismo nombre + versión
    for (const auto& m : models) {
        if (m.model_name == metadata.model_name && m.version == metadata.version) {
            throw std::runtime_error(
                "Model already exists: " + metadata.model_name +
                " v" + std::to_string(metadata.version)
            );
        }
    }

    models.push_back(metadata);
    save_to_disk();
}

std::vector<ModelCatalog::ModelMetadata> ModelCatalog::list_models() const {
    return models;
}

ModelCatalog::ModelMetadata ModelCatalog::get_model(
    const std::string& model_name,
    int version
) const {
    if (version == -1) {
        // Obtener última versión
        version = get_latest_version(model_name);
    }

    for (const auto& m : models) {
        if (m.model_name == model_name && m.version == version) {
            return m;
        }
    }

    throw std::runtime_error("Model not found: " + model_name + " v" + std::to_string(version));
}

bool ModelCatalog::model_exists(const std::string& model_name, int version) const {
    try {
        get_model(model_name, version);
        return true;
    } catch (...) {
        return false;
    }
}

int ModelCatalog::get_latest_version(const std::string& model_name) const {
    int max_version = -1;

    for (const auto& m : models) {
        if (m.model_name == model_name && m.version > max_version) {
            max_version = m.version;
        }
    }

    if (max_version == -1) {
        throw std::runtime_error("Model not found: " + model_name);
    }

    return max_version;
}

void ModelCatalog::delete_model(const std::string& model_name, int version) {
    auto it = std::remove_if(models.begin(), models.end(), [&](const ModelMetadata& m) {
        return m.model_name == model_name && m.version == version;
    });

    if (it == models.end()) {
        throw std::runtime_error("Model not found: " + model_name + " v" + std::to_string(version));
    }

    // Borrar archivos de disco
    auto metadata = *it;
    std::filesystem::remove(metadata.weights_path);
    std::filesystem::remove(metadata.config_path);

    models.erase(it, models.end());
    save_to_disk();
}

void ModelCatalog::load_from_disk() {
    std::ifstream file(catalog_path);
    if (!file.is_open()) {
        // Catálogo vacío
        return;
    }

    json catalog_json;
    file >> catalog_json;

    for (const auto& model_json : catalog_json["models"]) {
        ModelMetadata metadata;
        metadata.model_id = model_json["model_id"];
        metadata.model_name = model_json["model_name"];
        metadata.version = model_json["version"];
        metadata.architecture = model_json["architecture"];
        metadata.task = model_json["task"];
        metadata.hidden_dims = model_json["hidden_dims"].get<std::vector<int>>();
        metadata.input_dim = model_json["input_dim"];
        metadata.output_dim = model_json["output_dim"];
        metadata.hyperparameters = model_json["hyperparameters"];
        metadata.node_features = model_json["node_features"].get<std::vector<std::string>>();
        metadata.label_property = model_json["label_property"];
        metadata.best_val_loss = model_json["best_val_loss"];
        metadata.best_val_accuracy = model_json["best_val_accuracy"];
        metadata.best_epoch = model_json["best_epoch"];
        metadata.created_at = model_json["created_at"];
        metadata.projection_name = model_json["projection_name"];
        metadata.projection_snapshot = model_json["projection_snapshot"];
        metadata.weights_path = model_json["weights_path"];
        metadata.config_path = model_json["config_path"];

        models.push_back(metadata);
    }
}

void ModelCatalog::save_to_disk() {
    json catalog_json;
    catalog_json["models"] = json::array();

    for (const auto& m : models) {
        json model_json;
        model_json["model_id"] = m.model_id;
        model_json["model_name"] = m.model_name;
        model_json["version"] = m.version;
        model_json["architecture"] = m.architecture;
        model_json["task"] = m.task;
        model_json["hidden_dims"] = m.hidden_dims;
        model_json["input_dim"] = m.input_dim;
        model_json["output_dim"] = m.output_dim;
        model_json["hyperparameters"] = m.hyperparameters;
        model_json["node_features"] = m.node_features;
        model_json["label_property"] = m.label_property;
        model_json["best_val_loss"] = m.best_val_loss;
        model_json["best_val_accuracy"] = m.best_val_accuracy;
        model_json["best_epoch"] = m.best_epoch;
        model_json["created_at"] = m.created_at;
        model_json["projection_name"] = m.projection_name;
        model_json["projection_snapshot"] = m.projection_snapshot;
        model_json["weights_path"] = m.weights_path;
        model_json["config_path"] = m.config_path;

        catalog_json["models"].push_back(model_json);
    }

    std::ofstream file(catalog_path);
    file << catalog_json.dump(2);
}

} // namespace ML
} // namespace GQL
```

---

### 3.2 GNN Model Wrapper (LibTorch)

**Archivo**: `src/graph_models/gql/ml/gnn_model.h`

```cpp
// src/graph_models/gql/ml/gnn_model.h
#pragma once

#include <torch/torch.h>
#include <memory>
#include <vector>
#include <string>

namespace GQL {
namespace ML {

// Clase base para todas las arquitecturas GNN
class GNNModel : public torch::nn::Module {
public:
    virtual ~GNNModel() = default;

    // Forward pass
    virtual torch::Tensor forward(
        torch::Tensor x,              // Node features [num_nodes, input_dim]
        torch::Tensor edge_index      // Edge list [2, num_edges]
    ) = 0;

    // Guardar modelo a disco
    virtual void save(const std::string& path) = 0;

    // Cargar modelo desde disco
    virtual void load(const std::string& path) = 0;

    // Metadata
    virtual std::string architecture_name() const = 0;
    virtual int input_dim() const = 0;
    virtual int output_dim() const = 0;
};

// ============================================================
// GCN (Graph Convolutional Network)
// ============================================================
class GCNModel : public GNNModel {
public:
    GCNModel(int input_dim, const std::vector<int>& hidden_dims, int output_dim);

    torch::Tensor forward(torch::Tensor x, torch::Tensor edge_index) override;

    void save(const std::string& path) override;
    void load(const std::string& path) override;

    std::string architecture_name() const override { return "GCN"; }
    int input_dim() const override { return input_dim_; }
    int output_dim() const override { return output_dim_; }

private:
    int input_dim_;
    int output_dim_;
    std::vector<int> hidden_dims_;

    // Capas
    std::vector<torch::nn::Linear> layers_;
};

// ============================================================
// GAT (Graph Attention Network)
// ============================================================
class GATModel : public GNNModel {
public:
    GATModel(
        int input_dim,
        const std::vector<int>& hidden_dims,
        int output_dim,
        int num_heads = 8  // Número de attention heads
    );

    torch::Tensor forward(torch::Tensor x, torch::Tensor edge_index) override;

    void save(const std::string& path) override;
    void load(const std::string& path) override;

    std::string architecture_name() const override { return "GAT"; }
    int input_dim() const override { return input_dim_; }
    int output_dim() const override { return output_dim_; }

private:
    int input_dim_;
    int output_dim_;
    std::vector<int> hidden_dims_;
    int num_heads_;

    // Capas de atención
    // (Implementación simplificada, en producción usar torch-geometric)
};

// ============================================================
// GraphSAGE
// ============================================================
class GraphSAGEModel : public GNNModel {
public:
    enum class Aggregator {
        MEAN,
        POOL,
        LSTM
    };

    GraphSAGEModel(
        int input_dim,
        const std::vector<int>& hidden_dims,
        int output_dim,
        Aggregator aggregator = Aggregator::MEAN
    );

    torch::Tensor forward(torch::Tensor x, torch::Tensor edge_index) override;

    void save(const std::string& path) override;
    void load(const std::string& path) override;

    std::string architecture_name() const override { return "GraphSAGE"; }
    int input_dim() const override { return input_dim_; }
    int output_dim() const override { return output_dim_; }

private:
    int input_dim_;
    int output_dim_;
    std::vector<int> hidden_dims_;
    Aggregator aggregator_;
};

// ============================================================
// Model Factory
// ============================================================
class GNNModelFactory {
public:
    static std::shared_ptr<GNNModel> create(
        const std::string& architecture,
        int input_dim,
        const std::vector<int>& hidden_dims,
        int output_dim,
        const nlohmann::json& config = {}
    );
};

} // namespace ML
} // namespace GQL
```

**Implementación simplificada de GCN**:

```cpp
// src/graph_models/gql/ml/gnn_model.cc
#include "gnn_model.h"
#include <torch/torch.h>

namespace GQL {
namespace ML {

// ============================================================
// GCN Implementation
// ============================================================
GCNModel::GCNModel(int input_dim, const std::vector<int>& hidden_dims, int output_dim)
    : input_dim_(input_dim), output_dim_(output_dim), hidden_dims_(hidden_dims)
{
    // Construir capas
    int prev_dim = input_dim;

    for (size_t i = 0; i < hidden_dims.size(); ++i) {
        int hidden_dim = hidden_dims[i];

        auto layer = torch::nn::Linear(prev_dim, hidden_dim);
        layers_.push_back(register_module("layer" + std::to_string(i), layer));

        prev_dim = hidden_dim;
    }

    // Capa de salida
    auto output_layer = torch::nn::Linear(prev_dim, output_dim);
    layers_.push_back(register_module("output", output_layer));
}

torch::Tensor GCNModel::forward(torch::Tensor x, torch::Tensor edge_index) {
    // Implementación simplificada de GCN
    // En producción, usar torch_geometric o implementar message passing completo

    // Normalización de adyacencia (simplificado)
    // A_norm = D^{-1/2} * A * D^{-1/2}

    int num_nodes = x.size(0);

    // Construir matriz de adyacencia sparse desde edge_index
    auto edge_index_cpu = edge_index.cpu();
    auto edge_index_accessor = edge_index_cpu.accessor<int64_t, 2>();

    std::vector<int64_t> indices_i, indices_j;
    std::vector<float> values;

    for (int64_t i = 0; i < edge_index.size(1); ++i) {
        int64_t src = edge_index_accessor[0][i];
        int64_t dst = edge_index_accessor[1][i];

        indices_i.push_back(src);
        indices_j.push_back(dst);
        values.push_back(1.0f);
    }

    // Crear matriz sparse
    auto indices = torch::stack({
        torch::tensor(indices_i, torch::kLong),
        torch::tensor(indices_j, torch::kLong)
    });

    auto values_tensor = torch::tensor(values);
    auto adj_matrix = torch::sparse_coo_tensor(
        indices,
        values_tensor,
        {num_nodes, num_nodes}
    );

    // Agregar self-loops
    auto self_loop_indices = torch::arange(num_nodes);
    auto self_loop_stack = torch::stack({self_loop_indices, self_loop_indices});
    auto self_loop_values = torch::ones(num_nodes);

    auto self_loop_matrix = torch::sparse_coo_tensor(
        self_loop_stack,
        self_loop_values,
        {num_nodes, num_nodes}
    );

    adj_matrix = adj_matrix + self_loop_matrix;

    // Normalización de grado (simplificada)
    auto degrees = torch::sparse::sum(adj_matrix, 1).to_dense() + 1e-6;
    auto degree_inv_sqrt = torch::pow(degrees, -0.5);

    // Forward pass por capas
    torch::Tensor h = x;

    for (size_t i = 0; i < layers_.size() - 1; ++i) {
        // Agregación de vecinos
        h = torch::sparse::mm(adj_matrix, h);

        // Normalización
        h = h * degree_inv_sqrt.unsqueeze(1);

        // Linear transformation
        h = layers_[i]->forward(h);

        // Activación ReLU
        h = torch::relu(h);

        // Dropout (solo en training)
        if (is_training()) {
            h = torch::dropout(h, 0.5, true);
        }
    }

    // Última capa sin ReLU
    h = torch::sparse::mm(adj_matrix, h);
    h = h * degree_inv_sqrt.unsqueeze(1);
    h = layers_.back()->forward(h);

    return h;
}

void GCNModel::save(const std::string& path) {
    torch::save(shared_from_this(), path);
}

void GCNModel::load(const std::string& path) {
    torch::load(shared_from_this(), path);
}

// ============================================================
// Factory
// ============================================================
std::shared_ptr<GNNModel> GNNModelFactory::create(
    const std::string& architecture,
    int input_dim,
    const std::vector<int>& hidden_dims,
    int output_dim,
    const nlohmann::json& config
) {
    if (architecture == "GCN") {
        return std::make_shared<GCNModel>(input_dim, hidden_dims, output_dim);
    } else if (architecture == "GAT") {
        int num_heads = config.value("num_heads", 8);
        return std::make_shared<GATModel>(input_dim, hidden_dims, output_dim, num_heads);
    } else if (architecture == "GraphSAGE") {
        // Parse aggregator
        std::string agg_str = config.value("aggregator", "mean");
        GraphSAGEModel::Aggregator agg = GraphSAGEModel::Aggregator::MEAN;
        if (agg_str == "pool") agg = GraphSAGEModel::Aggregator::POOL;
        else if (agg_str == "lstm") agg = GraphSAGEModel::Aggregator::LSTM;

        return std::make_shared<GraphSAGEModel>(input_dim, hidden_dims, output_dim, agg);
    } else {
        throw std::runtime_error("Unknown architecture: " + architecture);
    }
}

} // namespace ML
} // namespace GQL
```

---

### 3.3 Training Pipeline

**Archivo**: `src/graph_models/gql/ml/trainer.h`

```cpp
// src/graph_models/gql/ml/trainer.h
#pragma once

#include "gnn_model.h"
#include "model_catalog.h"
#include <torch/torch.h>
#include <memory>
#include <functional>

namespace GQL {
namespace ML {

class GNNTrainer {
public:
    struct TrainingConfig {
        int epochs = 200;
        float learning_rate = 0.01;
        int batch_size = 1024;

        bool early_stopping = true;
        int patience = 10;              // Epochs sin mejora antes de parar

        bool use_gpu = true;
        int checkpoint_interval = 10;   // Guardar checkpoint cada N epochs

        std::string optimizer_type = "adam";  // "adam", "sgd"
        float weight_decay = 0.0f;

        // Callbacks
        std::function<void(int epoch, float train_loss, float val_loss, float val_acc)>
            progress_callback;
    };

    struct TrainingData {
        torch::Tensor x;              // Node features [num_nodes, input_dim]
        torch::Tensor edge_index;     // Edge list [2, num_edges]
        torch::Tensor y;              // Labels [num_nodes]

        torch::Tensor train_mask;     // Boolean mask [num_nodes]
        torch::Tensor val_mask;       // Boolean mask [num_nodes]
        torch::Tensor test_mask;      // Boolean mask [num_nodes]
    };

    struct TrainingResult {
        float best_val_loss;
        float best_val_accuracy;
        int best_epoch;

        std::vector<float> train_loss_history;
        std::vector<float> val_loss_history;
        std::vector<float> val_acc_history;
    };

    GNNTrainer(
        std::shared_ptr<GNNModel> model,
        const TrainingConfig& config
    );

    // Entrenar modelo
    TrainingResult train(const TrainingData& data);

    // Evaluar modelo
    float evaluate(const TrainingData& data, const torch::Tensor& mask);

private:
    std::shared_ptr<GNNModel> model_;
    TrainingConfig config_;
    std::unique_ptr<torch::optim::Optimizer> optimizer_;
    torch::Device device_;

    void setup_optimizer();
    float compute_loss(torch::Tensor output, torch::Tensor labels, torch::Tensor mask);
    float compute_accuracy(torch::Tensor output, torch::Tensor labels, torch::Tensor mask);
};

} // namespace ML
} // namespace GQL
```

**Implementación**:

```cpp
// src/graph_models/gql/ml/trainer.cc
#include "trainer.h"
#include <iostream>

namespace GQL {
namespace ML {

GNNTrainer::GNNTrainer(
    std::shared_ptr<GNNModel> model,
    const TrainingConfig& config
)
    : model_(model), config_(config)
{
    // Seleccionar dispositivo (GPU o CPU)
    if (config_.use_gpu && torch::cuda::is_available()) {
        device_ = torch::Device(torch::kCUDA);
        std::cout << "Using GPU for training" << std::endl;
    } else {
        device_ = torch::Device(torch::kCPU);
        std::cout << "Using CPU for training" << std::endl;
    }

    // Mover modelo a dispositivo
    model_->to(device_);

    setup_optimizer();
}

void GNNTrainer::setup_optimizer() {
    if (config_.optimizer_type == "adam") {
        optimizer_ = std::make_unique<torch::optim::Adam>(
            model_->parameters(),
            torch::optim::AdamOptions(config_.learning_rate)
                .weight_decay(config_.weight_decay)
        );
    } else if (config_.optimizer_type == "sgd") {
        optimizer_ = std::make_unique<torch::optim::SGD>(
            model_->parameters(),
            torch::optim::SGDOptions(config_.learning_rate)
                .weight_decay(config_.weight_decay)
                .momentum(0.9)
        );
    } else {
        throw std::runtime_error("Unknown optimizer: " + config_.optimizer_type);
    }
}

GNNTrainer::TrainingResult GNNTrainer::train(const TrainingData& data) {
    TrainingResult result;

    // Mover datos a dispositivo
    auto x = data.x.to(device_);
    auto edge_index = data.edge_index.to(device_);
    auto y = data.y.to(device_);
    auto train_mask = data.train_mask.to(device_);
    auto val_mask = data.val_mask.to(device_);

    float best_val_loss = std::numeric_limits<float>::infinity();
    int patience_counter = 0;

    std::cout << "Starting training..." << std::endl;
    std::cout << "  Epochs: " << config_.epochs << std::endl;
    std::cout << "  Learning rate: " << config_.learning_rate << std::endl;
    std::cout << "  Batch size: " << config_.batch_size << std::endl;

    for (int epoch = 0; epoch < config_.epochs; ++epoch) {
        // ============================================================
        // TRAINING STEP
        // ============================================================
        model_->train();
        optimizer_->zero_grad();

        // Forward pass
        auto output = model_->forward(x, edge_index);

        // Compute loss (solo en nodos de entrenamiento)
        float train_loss = compute_loss(output, y, train_mask);

        // Backward pass
        auto loss_tensor = torch::tensor(train_loss, torch::requires_grad(true));
        loss_tensor.backward();

        // Update weights
        optimizer_->step();

        // ============================================================
        // VALIDATION STEP
        // ============================================================
        model_->eval();
        torch::NoGradGuard no_grad;

        auto val_output = model_->forward(x, edge_index);
        float val_loss = compute_loss(val_output, y, val_mask);
        float val_acc = compute_accuracy(val_output, y, val_mask);

        // Guardar métricas
        result.train_loss_history.push_back(train_loss);
        result.val_loss_history.push_back(val_loss);
        result.val_acc_history.push_back(val_acc);

        // Early stopping
        if (val_loss < best_val_loss) {
            best_val_loss = val_loss;
            result.best_val_loss = val_loss;
            result.best_val_accuracy = val_acc;
            result.best_epoch = epoch;
            patience_counter = 0;
        } else {
            patience_counter++;
        }

        // Callback de progreso
        if (config_.progress_callback) {
            config_.progress_callback(epoch, train_loss, val_loss, val_acc);
        }

        // Print cada 10 epochs
        if (epoch % 10 == 0) {
            std::cout << "Epoch " << epoch
                      << " | Train Loss: " << train_loss
                      << " | Val Loss: " << val_loss
                      << " | Val Acc: " << val_acc
                      << std::endl;
        }

        // Early stopping
        if (config_.early_stopping && patience_counter >= config_.patience) {
            std::cout << "Early stopping at epoch " << epoch << std::endl;
            break;
        }
    }

    std::cout << "Training completed!" << std::endl;
    std::cout << "  Best val loss: " << result.best_val_loss << std::endl;
    std::cout << "  Best val accuracy: " << result.best_val_accuracy << std::endl;
    std::cout << "  Best epoch: " << result.best_epoch << std::endl;

    return result;
}

float GNNTrainer::compute_loss(
    torch::Tensor output,
    torch::Tensor labels,
    torch::Tensor mask
) {
    // Cross-entropy loss solo en nodos con mask = true
    auto masked_output = output.index({mask});
    auto masked_labels = labels.index({mask});

    auto loss = torch::nn::functional::cross_entropy(masked_output, masked_labels);
    return loss.item<float>();
}

float GNNTrainer::compute_accuracy(
    torch::Tensor output,
    torch::Tensor labels,
    torch::Tensor mask
) {
    auto predictions = output.argmax(1);
    auto masked_predictions = predictions.index({mask});
    auto masked_labels = labels.index({mask});

    auto correct = (masked_predictions == masked_labels).sum();
    auto total = mask.sum();

    return correct.item<float>() / total.item<float>();
}

float GNNTrainer::evaluate(const TrainingData& data, const torch::Tensor& mask) {
    model_->eval();
    torch::NoGradGuard no_grad;

    auto x = data.x.to(device_);
    auto edge_index = data.edge_index.to(device_);
    auto y = data.y.to(device_);
    auto eval_mask = mask.to(device_);

    auto output = model_->forward(x, edge_index);
    return compute_accuracy(output, y, eval_mask);
}

} // namespace ML
} // namespace GQL
```

---

### 3.4 Data Loader desde Proyección

**Archivo**: `src/graph_models/gql/ml/graph_data_loader.h`

```cpp
// src/graph_models/gql/ml/graph_data_loader.h
#pragma once

#include <torch/torch.h>
#include <string>
#include <vector>
#include "query/query_context.h"
#include "trainer.h"

namespace GQL {
namespace ML {

class GraphDataLoader {
public:
    struct DataConfig {
        std::vector<std::string> node_features;  // ["age", "pagerank", "embedding"]
        std::string label_property;              // "community"
        std::string train_mask_property;         // "is_train"
        std::string val_mask_property;           // "is_val"
        std::string test_mask_property;          // "is_test"
    };

    GraphDataLoader(const std::string& projection_name, const DataConfig& config);

    // Cargar datos completos en memoria (para grafos pequeños-medianos)
    GNNTrainer::TrainingData load_full_graph();

    // Cargar en batches (para grafos muy grandes)
    // TODO: Implementar en futuro para escalabilidad
    // std::vector<GNNTrainer::TrainingData> load_batches(int batch_size);

private:
    std::string projection_name_;
    DataConfig config_;

    // Helpers
    torch::Tensor load_node_features();
    torch::Tensor load_edge_index();
    torch::Tensor load_labels();
    torch::Tensor load_mask(const std::string& mask_property);

    int count_unique_labels();
    std::unordered_map<std::string, int> create_label_mapping();
};

} // namespace ML
} // namespace GQL
```

**Implementación**:

```cpp
// src/graph_models/gql/ml/graph_data_loader.cc
#include "graph_data_loader.h"
#include "graph_models/gql/gql_model.h"
#include "graph_models/conversions.h"

namespace GQL {
namespace ML {

GraphDataLoader::GraphDataLoader(
    const std::string& projection_name,
    const DataConfig& config
)
    : projection_name_(projection_name), config_(config)
{
}

GNNTrainer::TrainingData GraphDataLoader::load_full_graph() {
    std::cout << "Loading graph data from projection: " << projection_name_ << std::endl;

    GNNTrainer::TrainingData data;

    // Cargar features
    std::cout << "  Loading node features..." << std::endl;
    data.x = load_node_features();
    std::cout << "    Feature matrix shape: [" << data.x.size(0) << ", " << data.x.size(1) << "]" << std::endl;

    // Cargar topología
    std::cout << "  Loading graph topology..." << std::endl;
    data.edge_index = load_edge_index();
    std::cout << "    Edge index shape: [" << data.edge_index.size(0) << ", " << data.edge_index.size(1) << "]" << std::endl;

    // Cargar labels
    std::cout << "  Loading labels..." << std::endl;
    data.y = load_labels();
    std::cout << "    Labels shape: [" << data.y.size(0) << "]" << std::endl;
    std::cout << "    Number of classes: " << count_unique_labels() << std::endl;

    // Cargar máscaras
    std::cout << "  Loading train/val/test masks..." << std::endl;
    data.train_mask = load_mask(config_.train_mask_property);
    data.val_mask = load_mask(config_.val_mask_property);
    data.test_mask = load_mask(config_.test_mask_property);

    std::cout << "    Train nodes: " << data.train_mask.sum().item<int>() << std::endl;
    std::cout << "    Val nodes: " << data.val_mask.sum().item<int>() << std::endl;
    std::cout << "    Test nodes: " << data.test_mask.sum().item<int>() << std::endl;

    std::cout << "Data loading complete!" << std::endl;

    return data;
}

torch::Tensor GraphDataLoader::load_node_features() {
    auto& gql_model = GQL::get_model();
    auto& ctx = get_query_ctx();

    // Cargar proyección
    ctx.load_projection(projection_name_);

    // Obtener índice de nodos
    auto& node_index = gql_model.get_node_id_index();

    // Contar nodos
    bool interruption = false;
    auto it = node_index->get_range(&interruption, Record<1>{0}, Record<1>{UINT64_MAX});

    std::vector<ObjectId> node_ids;
    auto record = it.next();
    while (record != nullptr) {
        node_ids.push_back(ObjectId((*record)[0]));
        record = it.next();
    }

    int num_nodes = node_ids.size();
    int num_features = 0;

    // Calcular dimensión total de features
    for (const auto& feature_name : config_.node_features) {
        // TODO: Manejar features vectoriales (embeddings)
        // Por ahora, asumir features escalares
        num_features += 1;
    }

    // Crear tensor de features
    auto feature_matrix = torch::zeros({num_nodes, num_features});

    // Llenar features
    for (int node_idx = 0; node_idx < num_nodes; ++node_idx) {
        ObjectId node_id = node_ids[node_idx];

        int feature_offset = 0;

        for (const auto& feature_name : config_.node_features) {
            // Buscar propiedad en node_key_value
            ObjectId key_oid = Conversions::pack_string(feature_name);

            BptIter<3> prop_it = gql_model.get_node_key_value().get_range(
                &interruption,
                Record<3>{node_id.id, key_oid.id, 0},
                Record<3>{node_id.id, key_oid.id, UINT64_MAX}
            );

            auto prop_record = prop_it.next();
            if (prop_record != nullptr) {
                ObjectId value_oid((*prop_record)[2]);

                // Convertir a float
                auto type = GQL_OID::get_type(value_oid);
                float value = 0.0f;

                if (type == GQL_OID::Type::INT) {
                    value = static_cast<float>(Conversions::unpack_int(value_oid));
                } else if (type == GQL_OID::Type::FLOAT) {
                    value = Conversions::unpack_float(value_oid);
                }

                feature_matrix[node_idx][feature_offset] = value;
            }

            feature_offset += 1;
        }
    }

    return feature_matrix;
}

torch::Tensor GraphDataLoader::load_edge_index() {
    auto& gql_model = GQL::get_model();
    auto& ctx = get_query_ctx();

    ctx.load_projection(projection_name_);

    // Crear mapeo node_id -> índice (0-based)
    auto& node_index = gql_model.get_node_id_index();
    bool interruption = false;
    auto it = node_index->get_range(&interruption, Record<1>{0}, Record<1>{UINT64_MAX});

    std::unordered_map<uint64_t, int> node_id_to_idx;
    int idx = 0;
    auto record = it.next();
    while (record != nullptr) {
        uint64_t node_id = (*record)[0];
        node_id_to_idx[node_id] = idx++;
        record = it.next();
    }

    // Cargar aristas
    auto& edge_index = gql_model.get_from_to_edge();
    auto edge_it = edge_index.get_range(
        &interruption,
        Record<3>{0, 0, 0},
        Record<3>{UINT64_MAX, UINT64_MAX, UINT64_MAX}
    );

    std::vector<int64_t> sources, targets;

    auto edge_record = edge_it.next();
    while (edge_record != nullptr) {
        uint64_t from_id = (*edge_record)[0];
        uint64_t to_id = (*edge_record)[1];

        int from_idx = node_id_to_idx[from_id];
        int to_idx = node_id_to_idx[to_id];

        sources.push_back(from_idx);
        targets.push_back(to_idx);

        edge_record = edge_it.next();
    }

    // Crear edge_index tensor [2, num_edges]
    auto edge_index_tensor = torch::zeros({2, static_cast<int64_t>(sources.size())}, torch::kLong);

    for (size_t i = 0; i < sources.size(); ++i) {
        edge_index_tensor[0][i] = sources[i];
        edge_index_tensor[1][i] = targets[i];
    }

    return edge_index_tensor;
}

torch::Tensor GraphDataLoader::load_labels() {
    // Similar a load_node_features pero solo cargar la propiedad de label
    // Convertir labels de string a int si es necesario

    auto& gql_model = GQL::get_model();
    auto& ctx = get_query_ctx();

    ctx.load_projection(projection_name_);

    // Crear mapeo de labels a índices
    auto label_mapping = create_label_mapping();

    // Cargar nodos
    auto& node_index = gql_model.get_node_id_index();
    bool interruption = false;
    auto it = node_index->get_range(&interruption, Record<1>{0}, Record<1>{UINT64_MAX});

    std::vector<ObjectId> node_ids;
    auto record = it.next();
    while (record != nullptr) {
        node_ids.push_back(ObjectId((*record)[0]));
        record = it.next();
    }

    int num_nodes = node_ids.size();
    auto labels = torch::zeros(num_nodes, torch::kLong);

    // Llenar labels
    ObjectId label_key_oid = Conversions::pack_string(config_.label_property);

    for (int node_idx = 0; node_idx < num_nodes; ++node_idx) {
        ObjectId node_id = node_ids[node_idx];

        BptIter<3> label_it = gql_model.get_node_key_value().get_range(
            &interruption,
            Record<3>{node_id.id, label_key_oid.id, 0},
            Record<3>{node_id.id, label_key_oid.id, UINT64_MAX}
        );

        auto label_record = label_it.next();
        if (label_record != nullptr) {
            ObjectId label_value_oid((*label_record)[2]);

            auto type = GQL_OID::get_type(label_value_oid);

            if (type == GQL_OID::Type::INT) {
                labels[node_idx] = Conversions::unpack_int(label_value_oid);
            } else if (type == GQL_OID::Type::STRING_SIMPLE_INLINE ||
                       type == GQL_OID::Type::STRING_SIMPLE_EXTERN) {
                std::string label_str = Conversions::unpack_string(label_value_oid);
                labels[node_idx] = label_mapping[label_str];
            }
        }
    }

    return labels;
}

torch::Tensor GraphDataLoader::load_mask(const std::string& mask_property) {
    auto& gql_model = GQL::get_model();
    auto& ctx = get_query_ctx();

    ctx.load_projection(projection_name_);

    // Cargar nodos
    auto& node_index = gql_model.get_node_id_index();
    bool interruption = false;
    auto it = node_index->get_range(&interruption, Record<1>{0}, Record<1>{UINT64_MAX});

    std::vector<ObjectId> node_ids;
    auto record = it.next();
    while (record != nullptr) {
        node_ids.push_back(ObjectId((*record)[0]));
        record = it.next();
    }

    int num_nodes = node_ids.size();
    auto mask = torch::zeros(num_nodes, torch::kBool);

    // Llenar mask
    ObjectId mask_key_oid = Conversions::pack_string(mask_property);

    for (int node_idx = 0; node_idx < num_nodes; ++node_idx) {
        ObjectId node_id = node_ids[node_idx];

        BptIter<3> mask_it = gql_model.get_node_key_value().get_range(
            &interruption,
            Record<3>{node_id.id, mask_key_oid.id, 0},
            Record<3>{node_id.id, mask_key_oid.id, UINT64_MAX}
        );

        auto mask_record = mask_it.next();
        if (mask_record != nullptr) {
            ObjectId mask_value_oid((*mask_record)[2]);
            bool mask_value = Conversions::unpack_bool(mask_value_oid);
            mask[node_idx] = mask_value;
        }
    }

    return mask;
}

int GraphDataLoader::count_unique_labels() {
    return create_label_mapping().size();
}

std::unordered_map<std::string, int> GraphDataLoader::create_label_mapping() {
    auto& gql_model = GQL::get_model();
    auto& ctx = get_query_ctx();

    ctx.load_projection(projection_name_);

    // Escanear todos los valores de label_property
    ObjectId label_key_oid = Conversions::pack_string(config_.label_property);

    bool interruption = false;
    BptIter<3> it = gql_model.get_key_value_node().get_range(
        &interruption,
        Record<3>{label_key_oid.id, 0, 0},
        Record<3>{label_key_oid.id, UINT64_MAX, UINT64_MAX}
    );

    std::set<std::string> unique_labels;

    auto record = it.next();
    while (record != nullptr) {
        ObjectId label_value_oid((*record)[1]);
        auto type = GQL_OID::get_type(label_value_oid);

        if (type == GQL_OID::Type::STRING_SIMPLE_INLINE ||
            type == GQL_OID::Type::STRING_SIMPLE_EXTERN) {
            std::string label_str = Conversions::unpack_string(label_value_oid);
            unique_labels.insert(label_str);
        }

        record = it.next();
    }

    // Crear mapeo
    std::unordered_map<std::string, int> label_mapping;
    int idx = 0;
    for (const auto& label : unique_labels) {
        label_mapping[label] = idx++;
    }

    return label_mapping;
}

} // namespace ML
} // namespace GQL
```

---

## 4. Sistema de Persistencia de Modelos

### 4.1 Estructura de Directorios

```
data/dbs/gql/<database>/projections/<projection_name>/
├── catalog.dat
├── node_id.{dir,leaf}
├── from_to_edge.{dir,leaf}
├── to_from_edge.{dir,leaf}
├── node_key_value.{dir,leaf}
├── key_value_node.{dir,leaf}
│
└── models/                              # NUEVO directorio para modelos GNN
    ├── model_catalog.json               # Catálogo de todos los modelos
    │
    ├── my_classifier_v1.pt              # Pesos del modelo (LibTorch format)
    ├── my_classifier_v1.json            # Configuración del modelo
    │
    ├── my_classifier_v2.pt
    ├── my_classifier_v2.json
    │
    ├── link_predictor_v1.pt
    ├── link_predictor_v1.json
    │
    └── checkpoints/                     # Checkpoints durante entrenamiento
        ├── my_classifier_v3_epoch_50.pt
        ├── my_classifier_v3_epoch_100.pt
        └── my_classifier_v3_best.pt
```

---

### 4.2 Formato de `model_catalog.json`

```json
{
  "models": [
    {
      "model_id": "my_classifier_v1",
      "model_name": "my_classifier",
      "version": 1,
      "architecture": "GCN",
      "task": "node_classification",

      "hidden_dims": [64, 32],
      "input_dim": 66,
      "output_dim": 10,

      "hyperparameters": {
        "epochs": 200,
        "learning_rate": 0.01,
        "batch_size": 1024,
        "optimizer": "adam",
        "weight_decay": 0.0001,
        "dropout": 0.5,
        "early_stopping": true,
        "patience": 10
      },

      "node_features": ["age", "pagerank", "degree"],
      "label_property": "community",

      "metrics": {
        "best_val_loss": 0.234,
        "best_val_accuracy": 0.892,
        "best_epoch": 143,
        "train_loss_final": 0.123,
        "test_accuracy": 0.887
      },

      "created_at": "2025-10-18T10:30:45Z",
      "projection_name": "social_network",
      "projection_snapshot": "sha256:abc123...",

      "weights_path": "models/my_classifier_v1.pt",
      "config_path": "models/my_classifier_v1.json"
    },
    {
      "model_id": "my_classifier_v2",
      "model_name": "my_classifier",
      "version": 2,
      "architecture": "GAT",
      "task": "node_classification",

      "hidden_dims": [128, 64, 32],
      "input_dim": 66,
      "output_dim": 10,

      "hyperparameters": {
        "epochs": 300,
        "learning_rate": 0.005,
        "num_heads": 8,
        "attention_dropout": 0.6
      },

      "node_features": ["age", "pagerank", "degree"],
      "label_property": "community",

      "metrics": {
        "best_val_loss": 0.189,
        "best_val_accuracy": 0.923,
        "best_epoch": 234,
        "test_accuracy": 0.918
      },

      "created_at": "2025-10-18T14:22:10Z",
      "projection_name": "social_network",
      "projection_snapshot": "sha256:abc123...",

      "weights_path": "models/my_classifier_v2.pt",
      "config_path": "models/my_classifier_v2.json"
    }
  ]
}
```

---

### 4.3 Formato de Configuración Individual (`.json`)

**Archivo**: `models/my_classifier_v1.json`

```json
{
  "model_id": "my_classifier_v1",
  "model_name": "my_classifier",
  "version": 1,

  "architecture": {
    "type": "GCN",
    "input_dim": 66,
    "hidden_dims": [64, 32],
    "output_dim": 10,
    "activation": "relu",
    "dropout": 0.5,
    "normalization": "batch_norm"
  },

  "training": {
    "task": "node_classification",
    "loss_function": "cross_entropy",
    "optimizer": {
      "type": "adam",
      "learning_rate": 0.01,
      "weight_decay": 0.0001,
      "betas": [0.9, 0.999]
    },
    "scheduler": {
      "type": "reduce_on_plateau",
      "factor": 0.5,
      "patience": 5
    },
    "epochs": 200,
    "batch_size": 1024,
    "early_stopping": {
      "enabled": true,
      "patience": 10,
      "min_delta": 0.001
    }
  },

  "data": {
    "projection_name": "social_network",
    "projection_snapshot": "sha256:abc123...",
    "node_features": [
      {"name": "age", "type": "float", "dim": 1},
      {"name": "pagerank", "type": "float", "dim": 1},
      {"name": "degree", "type": "float", "dim": 1}
    ],
    "label_property": "community",
    "num_classes": 10,
    "train_mask": "is_train",
    "val_mask": "is_val",
    "test_mask": "is_test",
    "train_nodes": 8000,
    "val_nodes": 1000,
    "test_nodes": 1000
  },

  "training_history": {
    "total_epochs": 143,
    "total_time_seconds": 3421,
    "best_epoch": 143,
    "train_loss": [0.89, 0.67, 0.54, "...", 0.123],
    "val_loss": [0.92, 0.71, 0.59, "...", 0.234],
    "val_accuracy": [0.67, 0.78, 0.83, "...", 0.892]
  },

  "evaluation": {
    "test_accuracy": 0.887,
    "test_loss": 0.241,
    "confusion_matrix": "[[...]]",
    "per_class_accuracy": [0.91, 0.88, 0.85, "..."]
  },

  "metadata": {
    "created_at": "2025-10-18T10:30:45Z",
    "created_by": "user123",
    "mdb_version": "1.0.0",
    "libtorch_version": "2.1.0",
    "device": "cuda:0",
    "seed": 42
  }
}
```

---

### 4.4 Versionado Automático

**Política de versionado**:
```
my_classifier_v1  → Primera versión
my_classifier_v2  → Nueva versión (arquitectura cambiada, hiperparámetros distintos)
my_classifier_v3  → Nueva versión (datos cambiados, proyección actualizada)
```

**Auto-incremento**:
```cpp
int ModelCatalog::get_next_version(const std::string& model_name) {
    int max_version = 0;
    for (const auto& m : models) {
        if (m.model_name == model_name && m.version > max_version) {
            max_version = m.version;
        }
    }
    return max_version + 1;
}
```

**Uso en GQL**:
```gql
-- Definir modelo sin especificar versión → auto-incrementa
CALL mdb.gnn.define("my_classifier", {...})
-- Internamente crea "my_classifier_v1"

-- Re-entrenar con distintos hiperparámetros
CALL mdb.gnn.define("my_classifier", {...})
-- Internamente crea "my_classifier_v2"

-- Usar última versión
CALL mdb.gnn.predict("my_classifier")
-- Internamente usa "my_classifier_v2" (última versión)

-- Usar versión específica
CALL mdb.gnn.predict("my_classifier", {version: 1})
-- Usa "my_classifier_v1"
```

---

### 4.5 Snapshots de Proyección

**Problema**: ¿Qué pasa si la proyección cambia (se actualizan datos) después de entrenar un modelo?

**Solución**: Snapshot del estado de la proyección

```cpp
class ProjectionSnapshot {
public:
    // Generar hash SHA256 del estado de la proyección
    static std::string compute_snapshot_hash(const std::string& projection_name) {
        // Hash de:
        // - Número de nodos
        // - Número de aristas
        // - Checksum de node_key_value (muestra)
        // - Checksum de from_to_edge (muestra)

        auto& gql_model = GQL::get_model();

        uint64_t num_nodes = count_nodes(projection_name);
        uint64_t num_edges = count_edges(projection_name);

        // Samplear 1000 nodos para checksum
        std::string sample_data = sample_node_properties(projection_name, 1000);

        // Combinar en string
        std::string data_to_hash =
            std::to_string(num_nodes) + ":" +
            std::to_string(num_edges) + ":" +
            sample_data;

        // Computar SHA256
        return compute_sha256(data_to_hash);
    }

    // Verificar si la proyección ha cambiado desde que se entrenó el modelo
    static bool verify_snapshot(
        const std::string& projection_name,
        const std::string& expected_hash
    ) {
        std::string current_hash = compute_snapshot_hash(projection_name);
        return current_hash == expected_hash;
    }
};
```

**Uso al entrenar**:
```cpp
// Al finalizar entrenamiento
std::string snapshot = ProjectionSnapshot::compute_snapshot_hash(projection_name);

ModelMetadata metadata;
metadata.projection_name = projection_name;
metadata.projection_snapshot = snapshot;

catalog.register_model(metadata);
```

**Uso al predecir**:
```cpp
// Al cargar modelo para inferencia
auto metadata = catalog.get_model("my_classifier");

bool valid = ProjectionSnapshot::verify_snapshot(
    metadata.projection_name,
    metadata.projection_snapshot
);

if (!valid) {
    std::cerr << "WARNING: Projection has changed since model was trained!" << std::endl;
    std::cerr << "Model may produce incorrect predictions." << std::endl;
    // Opcionalmente: lanzar error o continuar con warning
}
```

---

## 5. Implementación Detallada: Procedimientos GQL

### 5.1 Arquitectura de Procedimientos

**Ubicación en código**:
```
src/query/executor/binding_iter/procedures/gql/
├── procedure_gnn_define.h/cc           # CALL mdb.gnn.define()
├── procedure_gnn_configure_data.h/cc   # CALL mdb.gnn.configure_data()
├── procedure_gnn_train.h/cc            # CALL mdb.gnn.train()
├── procedure_gnn_predict.h/cc          # CALL mdb.gnn.predict()
├── procedure_gnn_evaluate.h/cc         # CALL mdb.gnn.evaluate()
├── procedure_gnn_list_models.h/cc      # CALL mdb.gnn.list_models()
└── procedure_gnn_delete_model.h/cc     # CALL mdb.gnn.delete_model()
```

**Registro en el sistema**:
```cpp
// src/query/executor/binding_iter/procedures/gql/procedure_registry.cc

void register_gnn_procedures() {
    ProcedureRegistry::register_procedure("mdb.gnn.define",
        std::make_unique<ProcedureGNNDefine>());

    ProcedureRegistry::register_procedure("mdb.gnn.configure_data",
        std::make_unique<ProcedureGNNConfigureData>());

    ProcedureRegistry::register_procedure("mdb.gnn.train",
        std::make_unique<ProcedureGNNTrain>());

    ProcedureRegistry::register_procedure("mdb.gnn.predict",
        std::make_unique<ProcedureGNNPredict>());

    ProcedureRegistry::register_procedure("mdb.gnn.evaluate",
        std::make_unique<ProcedureGNNEvaluate>());

    ProcedureRegistry::register_procedure("mdb.gnn.list_models",
        std::make_unique<ProcedureGNNListModels>());

    ProcedureRegistry::register_procedure("mdb.gnn.delete_model",
        std::make_unique<ProcedureGNNDeleteModel>());
}
```

---

### 5.2 CALL mdb.gnn.define()

**Propósito**: Definir arquitectura de un modelo GNN

**Sintaxis**:
```gql
CALL mdb.gnn.define(
    model_name: STRING,
    config: MAP
) YIELD model_id, version
```

**Ejemplo**:
```gql
CALL mdb.gnn.define("my_classifier", {
    architecture: "GCN",
    task: "node_classification",
    hidden_dims: [64, 32],
    dropout: 0.5
}) YIELD model_id, version
```

**Implementación**:
```cpp
// src/query/executor/binding_iter/procedures/gql/procedure_gnn_define.h
#pragma once

#include "query/executor/binding_iter.h"
#include "graph_models/gql/ml/model_catalog.h"

namespace GQL {

class ProcedureGNNDefine : public BindingIter {
public:
    void _begin(BindingId& out_binding) override;
    void _reset() override;
    bool _next() override;

private:
    std::string model_name_;
    nlohmann::json config_;

    bool yielded_ = false;
    std::string generated_model_id_;
    int generated_version_;
};

} // namespace GQL
```

```cpp
// src/query/executor/binding_iter/procedures/gql/procedure_gnn_define.cc
#include "procedure_gnn_define.h"
#include "graph_models/gql/ml/model_catalog.h"

namespace GQL {

void ProcedureGNNDefine::_begin(BindingId& out_binding) {
    // Parsear argumentos de la query
    // (Implementación dependiente del parser GQL)

    auto& ctx = get_query_ctx();
    std::string projection_name = ctx.get_current_projection();

    // Cargar catálogo
    ML::ModelCatalog catalog(get_projection_dir(projection_name));

    // Generar siguiente versión
    int next_version = catalog.get_latest_version(model_name_) + 1;

    // Crear metadata inicial
    ML::ModelCatalog::ModelMetadata metadata;
    metadata.model_name = model_name_;
    metadata.version = next_version;
    metadata.model_id = model_name_ + "_v" + std::to_string(next_version);
    metadata.architecture = config_["architecture"];
    metadata.task = config_["task"];
    metadata.hidden_dims = config_["hidden_dims"].get<std::vector<int>>();
    metadata.hyperparameters = config_;
    metadata.created_at = get_current_timestamp_iso8601();
    metadata.projection_name = projection_name;

    // Registrar (sin entrenar todavía)
    catalog.register_model(metadata);

    // Guardar para yield
    generated_model_id_ = metadata.model_id;
    generated_version_ = next_version;
}

bool ProcedureGNNDefine::_next() {
    if (yielded_) {
        return false;
    }

    yielded_ = true;

    // Yield model_id y version
    parent_binding->add(var_model_id, ObjectId(generated_model_id_));
    parent_binding->add(var_version, ObjectId(generated_version_));

    return true;
}

} // namespace GQL
```

---

### 5.3 CALL mdb.gnn.configure_data()

**Propósito**: Configurar features y labels para entrenamiento

**Sintaxis**:
```gql
CALL mdb.gnn.configure_data(
    model_name: STRING,
    config: MAP
) YIELD status
```

**Ejemplo**:
```gql
CALL mdb.gnn.configure_data("my_classifier", {
    node_features: ["age", "pagerank", "degree"],
    node_labels: "community",
    train_mask: "is_train",
    val_mask: "is_val",
    test_mask: "is_test"
}) YIELD status
```

**Implementación**: Similar a `define`, actualiza metadata con configuración de datos.

---

### 5.4 CALL mdb.gnn.train()

**Propósito**: Entrenar modelo GNN

**Sintaxis**:
```gql
CALL mdb.gnn.train(
    model_name: STRING,
    config: MAP = {}
) YIELD epoch, train_loss, val_loss, val_accuracy
```

**Ejemplo**:
```gql
-- Entrenar con configuración por defecto
CALL mdb.gnn.train("my_classifier")
YIELD epoch, train_loss, val_loss, val_accuracy

-- Entrenar con hiperparámetros custom
CALL mdb.gnn.train("my_classifier", {
    epochs: 300,
    learning_rate: 0.005,
    early_stopping: true,
    patience: 20
}) YIELD epoch, train_loss, val_loss, val_accuracy
```

**Implementación**:
```cpp
// src/query/executor/binding_iter/procedures/gql/procedure_gnn_train.cc
#include "procedure_gnn_train.h"
#include "graph_models/gql/ml/model_catalog.h"
#include "graph_models/gql/ml/gnn_model.h"
#include "graph_models/gql/ml/trainer.h"
#include "graph_models/gql/ml/graph_data_loader.h"

namespace GQL {

void ProcedureGNNTrain::_begin(BindingId& out_binding) {
    auto& ctx = get_query_ctx();
    std::string projection_name = ctx.get_current_projection();

    // Cargar metadata del modelo
    ML::ModelCatalog catalog(get_projection_dir(projection_name));
    auto metadata = catalog.get_model(model_name_);

    // Crear modelo
    auto model = ML::GNNModelFactory::create(
        metadata.architecture,
        metadata.input_dim,
        metadata.hidden_dims,
        metadata.output_dim
    );

    // Configurar trainer
    ML::GNNTrainer::TrainingConfig train_config;
    if (config_.contains("epochs")) {
        train_config.epochs = config_["epochs"];
    }
    if (config_.contains("learning_rate")) {
        train_config.learning_rate = config_["learning_rate"];
    }

    // Callback para yield de progreso
    train_config.progress_callback = [this](int epoch, float train_loss,
                                             float val_loss, float val_acc) {
        // Guardar métricas para yield
        progress_data_.push_back({epoch, train_loss, val_loss, val_acc});
    };

    // Crear trainer
    ML::GNNTrainer trainer(model, train_config);

    // Cargar datos
    ML::GraphDataLoader::DataConfig data_config;
    data_config.node_features = metadata.node_features;
    data_config.label_property = metadata.label_property;
    data_config.train_mask_property = "is_train";
    data_config.val_mask_property = "is_val";
    data_config.test_mask_property = "is_test";

    ML::GraphDataLoader loader(projection_name, data_config);
    auto training_data = loader.load_full_graph();

    // ENTRENAR (blocking call)
    auto result = trainer.train(training_data);

    // Guardar modelo a disco
    std::string weights_path = get_projection_dir(projection_name) +
                               "/models/" + metadata.model_id + ".pt";
    model->save(weights_path);

    // Actualizar metadata con resultados
    metadata.best_val_loss = result.best_val_loss;
    metadata.best_val_accuracy = result.best_val_accuracy;
    metadata.best_epoch = result.best_epoch;
    metadata.weights_path = weights_path;

    // Generar snapshot de la proyección
    metadata.projection_snapshot = ProjectionSnapshot::compute_snapshot_hash(projection_name);

    catalog.register_model(metadata); // Actualizar catálogo
}

bool ProcedureGNNTrain::_next() {
    if (current_progress_idx_ >= progress_data_.size()) {
        return false;
    }

    auto& progress = progress_data_[current_progress_idx_++];

    // Yield epoch, train_loss, val_loss, val_accuracy
    parent_binding->add(var_epoch, ObjectId(progress.epoch));
    parent_binding->add(var_train_loss, ObjectId(progress.train_loss));
    parent_binding->add(var_val_loss, ObjectId(progress.val_loss));
    parent_binding->add(var_val_accuracy, ObjectId(progress.val_accuracy));

    return true;
}

} // namespace GQL
```

---

### 5.5 CALL mdb.gnn.predict()

**Propósito**: Realizar inferencia con modelo entrenado

**Sintaxis (Modo 1 - Batch completo)**:
```gql
CALL mdb.gnn.predict(model_name: STRING)
YIELD nodeId, prediction, probability
[MUTATE PROPERTY predicted_class = prediction]
```

**Sintaxis (Modo 2 - Inline en query)**:
```gql
MATCH (u:User)
WHERE u.age > 30
WITH u
CALL mdb.gnn.predict(model_name: STRING, node: u)
YIELD prediction, probability
WHERE probability > 0.8
RETURN u.name, prediction, probability
```

**Ejemplo 1 - Batch**:
```gql
-- Predecir en todos los nodos y guardar como propiedad
CALL mdb.gnn.predict("my_classifier")
YIELD nodeId, prediction, probability
MUTATE PROPERTY predicted_class = prediction,
                predicted_prob = probability
```

**Ejemplo 2 - Inline**:
```gql
-- Predecir solo en subset de nodos
MATCH (u:User {country: "Chile"})
WHERE u.age > 25
WITH u
CALL mdb.gnn.predict("my_classifier", u)
YIELD prediction, probability
WHERE probability > 0.9
RETURN u.name, u.age, prediction, probability
ORDER BY probability DESC
LIMIT 10
```

**Implementación - Modo Batch**:
```cpp
void ProcedureGNNPredict::_begin_batch_mode(BindingId& out_binding) {
    auto& ctx = get_query_ctx();
    std::string projection_name = ctx.get_current_projection();

    // Cargar modelo
    ML::ModelCatalog catalog(get_projection_dir(projection_name));
    auto metadata = catalog.get_model(model_name_);

    // Verificar snapshot
    bool valid = ProjectionSnapshot::verify_snapshot(
        projection_name,
        metadata.projection_snapshot
    );
    if (!valid) {
        std::cerr << "WARNING: Projection changed since model training!" << std::endl;
    }

    // Cargar pesos
    auto model = ML::GNNModelFactory::create(
        metadata.architecture,
        metadata.input_dim,
        metadata.hidden_dims,
        metadata.output_dim
    );
    model->load(metadata.weights_path);
    model->eval(); // Modo evaluación

    // Cargar datos completos
    ML::GraphDataLoader::DataConfig data_config;
    data_config.node_features = metadata.node_features;

    ML::GraphDataLoader loader(projection_name, data_config);
    auto data = loader.load_full_graph();

    // Inferencia
    torch::NoGradGuard no_grad;
    auto output = model->forward(data.x, data.edge_index);

    // Convertir output a predictions
    auto predictions = output.argmax(1); // [num_nodes]
    auto probabilities = torch::softmax(output, 1); // [num_nodes, num_classes]
    auto max_probs = std::get<0>(probabilities.max(1)); // [num_nodes]

    // Guardar para yield
    predictions_ = predictions.cpu();
    probabilities_ = max_probs.cpu();

    // Cargar node IDs
    node_ids_ = loader.get_node_ids();
}

bool ProcedureGNNPredict::_next_batch_mode() {
    if (current_node_idx_ >= node_ids_.size()) {
        return false;
    }

    ObjectId node_id = node_ids_[current_node_idx_];
    int prediction = predictions_[current_node_idx_].item<int>();
    float probability = probabilities_[current_node_idx_].item<float>();

    current_node_idx_++;

    // Yield
    parent_binding->add(var_nodeId, node_id);
    parent_binding->add(var_prediction, ObjectId(prediction));
    parent_binding->add(var_probability, ObjectId(probability));

    return true;
}
```

**Implementación - Modo Inline**:
```cpp
bool ProcedureGNNPredict::_next_inline_mode() {
    // Recibir nodo desde pipeline anterior (MATCH)
    if (!parent->next()) {
        return false;
    }

    ObjectId node_id = parent_binding->get(input_var_node);

    // Cargar vecindad del nodo (k-hop)
    auto subgraph = extract_k_hop_subgraph(node_id, k_hops=2);

    // Inferencia en subgrafo
    torch::NoGradGuard no_grad;
    auto output = model_->forward(subgraph.x, subgraph.edge_index);

    // Extraer predicción del nodo target
    int target_idx = subgraph.node_id_to_idx[node_id.id];
    auto node_output = output[target_idx];

    int prediction = node_output.argmax(0).item<int>();
    auto probs = torch::softmax(node_output, 0);
    float probability = probs[prediction].item<float>();

    // Yield
    parent_binding->add(var_prediction, ObjectId(prediction));
    parent_binding->add(var_probability, ObjectId(probability));

    return true;
}
```

---

### 5.6 Otros Procedimientos

**CALL mdb.gnn.evaluate()**:
```gql
-- Evaluar modelo en test set
CALL mdb.gnn.evaluate("my_classifier")
YIELD test_accuracy, test_loss, per_class_accuracy
```

**CALL mdb.gnn.list_models()**:
```gql
-- Listar todos los modelos
CALL mdb.gnn.list_models()
YIELD model_name, version, architecture, test_accuracy, created_at

-- Filtrar por nombre
CALL mdb.gnn.list_models({model_name: "my_classifier"})
YIELD version, test_accuracy, created_at
ORDER BY version DESC
```

**CALL mdb.gnn.delete_model()**:
```gql
-- Borrar versión específica
CALL mdb.gnn.delete_model("my_classifier", {version: 1})
YIELD status

-- Borrar todas las versiones
CALL mdb.gnn.delete_model("my_classifier", {all_versions: true})
YIELD status
```

---

## 6. Ejemplos de Uso End-to-End

### 6.1 Ejemplo Completo: Clasificación de Nodos en Red Social

**Escenario**: Clasificar usuarios de una red social en comunidades usando GNN

**Paso 1: Importar datos y crear proyección**

```bash
# Importar grafo principal
build/Release/bin/mdb import data/example/gql/social_network/network.gql social_network_db

# Iniciar servidor
build/Release/bin/mdb server social_network_db
```

```gql
-- Crear proyección con estructura, etiquetas y propiedades
MATCH (n)-[e]->(m)
RETURN PROJECT("full_network" INCLUDE LABELS INCLUDE PROPERTIES)
```

**Resultado**:
```
Projection created: full_network
Nodes: 10,000
Edges: 50,000
Size on disk: 245 MB
```

---

**Paso 2: Feature Engineering**

```gql
-- Usar proyección
USE "full_network"

-- Calcular PageRank (asumiendo que existe procedimiento)
CALL mdb.analytics.pagerank()
YIELD nodeId, pagerank
MUTATE PROPERTY pagerank = pagerank

-- Calcular grado
MATCH (n)
WITH n, count{(n)-[]->()} AS out_degree
MUTATE PROPERTY degree = out_degree

-- Calcular edad normalizada
MATCH (n)
WHERE n.age IS NOT NULL
WITH n, n.age / 100.0 AS age_normalized
MUTATE PROPERTY age_norm = age_normalized
```

**Resultado**:
```
Properties added:
  - pagerank: 10,000 nodes
  - degree: 10,000 nodes
  - age_norm: 10,000 nodes
```

---

**Paso 3: Crear train/val/test splits**

```gql
-- Split 80/10/10
MATCH (n)
WITH n, rand() AS r
MUTATE PROPERTY
  is_train = CASE WHEN r < 0.8 THEN true ELSE false END,
  is_val = CASE WHEN r >= 0.8 AND r < 0.9 THEN true ELSE false END,
  is_test = CASE WHEN r >= 0.9 THEN true ELSE false END
```

**Resultado**:
```
Train nodes: 8,000
Val nodes: 1,000
Test nodes: 1,000
```

---

**Paso 4: Definir modelo GNN**

```gql
-- Definir arquitectura GCN
CALL mdb.gnn.define("community_classifier", {
    architecture: "GCN",
    task: "node_classification",
    hidden_dims: [128, 64],
    dropout: 0.5
}) YIELD model_id, version

-- Configurar datos de entrenamiento
CALL mdb.gnn.configure_data("community_classifier", {
    node_features: ["age_norm", "pagerank", "degree"],
    node_labels: "community",
    train_mask: "is_train",
    val_mask: "is_val",
    test_mask: "is_test"
}) YIELD status
```

**Resultado**:
```
model_id: community_classifier_v1
version: 1
status: configured
Input dim: 3 (age_norm + pagerank + degree)
Output dim: 5 (5 communities detected)
```

---

**Paso 5: Entrenar modelo**

```gql
-- Entrenar con early stopping
CALL mdb.gnn.train("community_classifier", {
    epochs: 200,
    learning_rate: 0.01,
    early_stopping: true,
    patience: 10
}) YIELD epoch, train_loss, val_loss, val_accuracy
```

**Resultado (últimas 10 epochs)**:
```
epoch | train_loss | val_loss | val_accuracy
------|------------|----------|-------------
140   | 0.245      | 0.312    | 0.876
141   | 0.242      | 0.310    | 0.878
142   | 0.239      | 0.308    | 0.881
143   | 0.236      | 0.306    | 0.883
144   | 0.233      | 0.304    | 0.885
145   | 0.230      | 0.302    | 0.887
146   | 0.227      | 0.300    | 0.889
147   | 0.224      | 0.298    | 0.891
148   | 0.221      | 0.296    | 0.893
149   | 0.218      | 0.295    | 0.895

Early stopping triggered at epoch 149
Best val accuracy: 0.895 (epoch 149)
Model saved: models/community_classifier_v1.pt
```

---

**Paso 6: Evaluar en test set**

```gql
-- Evaluar rendimiento final
CALL mdb.gnn.evaluate("community_classifier")
YIELD test_accuracy, test_loss, per_class_accuracy
```

**Resultado**:
```
test_accuracy: 0.892
test_loss: 0.301
per_class_accuracy:
  Community 0: 0.91
  Community 1: 0.88
  Community 2: 0.89
  Community 3: 0.87
  Community 4: 0.90
```

---

**Paso 7: Inferencia en producción**

```gql
-- Opción A: Batch prediction (todos los nodos)
CALL mdb.gnn.predict("community_classifier")
YIELD nodeId, prediction, probability
MUTATE PROPERTY predicted_community = prediction,
                prediction_confidence = probability

-- Verificar
MATCH (n)
WHERE n.predicted_community IS NOT NULL
RETURN n.predicted_community, count(n) AS num_users
ORDER BY num_users DESC
```

**Resultado**:
```
predicted_community | num_users
--------------------|----------
0                   | 2,345
1                   | 2,123
2                   | 2,001
3                   | 1,890
4                   | 1,641
```

```gql
-- Opción B: Inline prediction (subset específico)
MATCH (u:User {country: "Chile"})
WHERE u.age > 30
WITH u
CALL mdb.gnn.predict("community_classifier", u)
YIELD prediction, probability
WHERE probability > 0.9  -- Solo predicciones con alta confianza
RETURN u.name, u.age, prediction AS community, probability
ORDER BY probability DESC
LIMIT 10
```

**Resultado**:
```
name          | age | community | probability
--------------|-----|-----------|------------
Juan Pérez    | 35  | 2         | 0.967
María García  | 42  | 0         | 0.954
Pedro López   | 38  | 2         | 0.943
Ana Silva     | 45  | 1         | 0.932
...
```

---

**Paso 8: Comparar con nueva versión (experimento)**

```gql
-- Definir nuevo modelo con arquitectura GAT
CALL mdb.gnn.define("community_classifier", {
    architecture: "GAT",
    task: "node_classification",
    hidden_dims: [256, 128, 64],
    num_heads: 8,
    dropout: 0.6
}) YIELD model_id, version

-- Configurar datos (mismo que antes)
CALL mdb.gnn.configure_data("community_classifier", {
    node_features: ["age_norm", "pagerank", "degree"],
    node_labels: "community",
    train_mask: "is_train",
    val_mask: "is_val",
    test_mask: "is_test"
}) YIELD status

-- Entrenar
CALL mdb.gnn.train("community_classifier")
YIELD epoch, train_loss, val_loss, val_accuracy
```

**Resultado**:
```
model_id: community_classifier_v2
version: 2
Best val accuracy: 0.912 (better than v1!)
Test accuracy: 0.908
```

```gql
-- Comparar modelos
CALL mdb.gnn.list_models({model_name: "community_classifier"})
YIELD version, architecture, test_accuracy, created_at
ORDER BY version DESC
```

**Resultado**:
```
version | architecture | test_accuracy | created_at
--------|--------------|---------------|-------------------
2       | GAT          | 0.908         | 2025-10-18T15:30:00Z
1       | GCN          | 0.892         | 2025-10-18T10:45:00Z
```

**Conclusión**: GAT v2 es mejor → usar en producción

```gql
-- Usar nueva versión por defecto (última)
CALL mdb.gnn.predict("community_classifier")  -- Usa v2 automáticamente
YIELD nodeId, prediction, probability
MUTATE PROPERTY predicted_community = prediction
```

---

### 6.2 Ejemplo: Link Prediction en Grafo de Colaboraciones

**Escenario**: Predecir futuras colaboraciones entre investigadores

**Datos**:
- Nodos: Investigadores (10,000)
- Aristas: Co-autorías (30,000)
- Objetivo: Predecir qué pares de investigadores colaborarán en el futuro

**Paso 1: Crear proyección temporal**

```gql
-- Solo considerar colaboraciones hasta 2020 (train)
MATCH (a:Researcher)-[c:COAUTHOR]->(b:Researcher)
WHERE c.year <= 2020
RETURN PROJECT("collab_2020" INCLUDE LABELS INCLUDE PROPERTIES)
```

**Paso 2: Feature engineering**

```gql
USE "collab_2020"

-- Calcular embeddings estructurales
CALL mdb.analytics.node2vec({
    walk_length: 80,
    num_walks: 10,
    p: 1,
    q: 1,
    dimensions: 64
}) YIELD nodeId, embedding
MUTATE PROPERTY node2vec_embedding = embedding

-- Calcular métricas locales
MATCH (n)
WITH n, count{(n)-[]->()} AS num_collaborators
MUTATE PROPERTY num_collabs = num_collaborators
```

**Paso 3: Generar pares positivos y negativos**

```gql
-- Pares positivos: colaboraciones 2021-2023
MATCH (a:Researcher)-[c:COAUTHOR]->(b:Researcher)
WHERE c.year > 2020 AND c.year <= 2023
WITH a, b
MUTATE EDGE link_label = 1  -- Positivo (colaboraron)

-- Pares negativos: no colaboraron (samplear)
MATCH (a:Researcher), (b:Researcher)
WHERE NOT EXISTS {(a)-[:COAUTHOR]-(b)}
AND rand() < 0.01  -- Samplear 1%
WITH a, b
MUTATE EDGE link_label = 0  -- Negativo (no colaboraron)
```

**Paso 4: Entrenar modelo de link prediction**

```gql
CALL mdb.gnn.define("collab_predictor", {
    architecture: "GraphSAGE",
    task: "link_prediction",
    hidden_dims: [128, 64],
    aggregator: "mean"
}) YIELD model_id

CALL mdb.gnn.configure_data("collab_predictor", {
    node_features: ["node2vec_embedding", "num_collabs"],
    edge_labels: "link_label",
    train_mask: "is_train_edge",
    val_mask: "is_val_edge"
}) YIELD status

CALL mdb.gnn.train("collab_predictor", {
    epochs: 150,
    learning_rate: 0.005
}) YIELD epoch, train_loss, val_loss, val_accuracy
```

**Resultado**:
```
Best val AUC: 0.87
Test AUC: 0.85
```

**Paso 5: Predecir futuras colaboraciones**

```gql
-- Predecir top 100 pares más probables
MATCH (a:Researcher), (b:Researcher)
WHERE a.id < b.id  -- Evitar duplicados
AND NOT EXISTS {(a)-[:COAUTHOR]-(b)}  -- Nunca han colaborado
WITH a, b
CALL mdb.gnn.predict_link("collab_predictor", a, b)
YIELD probability
WHERE probability > 0.8
RETURN a.name, b.name, probability
ORDER BY probability DESC
LIMIT 100
```

**Resultado**:
```
researcher_a       | researcher_b       | probability
-------------------|--------------------|-----------
Dr. Smith          | Dr. Johnson        | 0.94
Dr. García         | Dr. López          | 0.91
Dr. Chen           | Dr. Wang           | 0.89
...
```

---

### 6.3 Ejemplo: Consultas Híbridas (Graph Query + ML)

**Caso de uso**: Encontrar usuarios influyentes de alta calidad en red social

**Query híbrida**:
```gql
USE "social_network"

-- Combinar filtros estructurales + predicción ML
MATCH (u:User)
WHERE u.country = "Chile"
  AND u.followers > 1000  -- Filtro estructural
WITH u

-- Predecir calidad de contenido (modelo pre-entrenado)
CALL mdb.gnn.predict("content_quality_model", u)
YIELD prediction, probability
WHERE prediction = "high_quality"
  AND probability > 0.9

-- Calcular influencia real (PageRank)
WITH u, probability
MATCH (u)<-[:FOLLOWS]-(follower)
WITH u, probability, count(follower) AS follower_count

-- Ordenar por combinación de métricas
RETURN u.name,
       u.username,
       follower_count,
       probability AS quality_score,
       follower_count * probability AS influence_score
ORDER BY influence_score DESC
LIMIT 20
```

**Explicación**:
- **Filtro estructural**: `country = "Chile"` y `followers > 1000`
- **Predicción ML**: Clasificar calidad de contenido con GNN
- **Agregación**: Combinar métricas estructurales (followers) con score ML
- **Resultado**: Top 20 usuarios influyentes de alta calidad

**Ventaja competitiva**: Esta query es **IMPOSIBLE** en sistemas tradicionales:
- Neo4j GDS: No permite predicción inline en queries Cypher
- PostgreSQL + ML: Requiere múltiples queries y joins costosos
- **MillenniumDB**: Todo en una sola query nativa

---

### 6.4 Ejemplo: Re-entrenamiento Incremental

**Escenario**: Grafo crece con el tiempo, necesitamos actualizar modelo

**Workflow**:

```gql
-- 1. Verificar estado actual del modelo
CALL mdb.gnn.list_models({model_name: "user_classifier"})
YIELD version, created_at, test_accuracy, projection_snapshot

-- Resultado:
-- version=1, created_at=2025-09-01, test_accuracy=0.88, snapshot=abc123

-- 2. Crear nueva proyección con datos actualizados
MATCH (n)-[e]->(m)
RETURN PROJECT("users_updated_oct" INCLUDE LABELS INCLUDE PROPERTIES)

-- 3. Cambiar a nueva proyección
USE "users_updated_oct"

-- 4. Re-entrenar modelo (crea versión 2 automáticamente)
CALL mdb.gnn.define("user_classifier", {
    architecture: "GCN",
    hidden_dims: [128, 64]
}) YIELD model_id, version
-- version=2

CALL mdb.gnn.configure_data("user_classifier", {...})
CALL mdb.gnn.train("user_classifier")
YIELD epoch, train_loss, val_loss, val_accuracy

-- Resultado:
-- Best val acc: 0.91 (mejor que v1!)

-- 5. Comparar versiones
CALL mdb.gnn.list_models({model_name: "user_classifier"})
YIELD version, test_accuracy, projection_snapshot
ORDER BY version DESC

-- version | test_accuracy | projection_snapshot
-- --------|---------------|--------------------
-- 2       | 0.91          | def456 (Oct 2025)
-- 1       | 0.88          | abc123 (Sep 2025)

-- 6. Usar nueva versión en producción
CALL mdb.gnn.predict("user_classifier")  -- Usa v2 automáticamente
YIELD nodeId, prediction, probability
MUTATE PROPERTY predicted_class = prediction
```

**Ventajas**:
- ✅ Versionado automático (v1, v2, v3...)
- ✅ Snapshots de proyección (trazabilidad)
- ✅ Comparación side-by-side
- ✅ Rollback fácil (usar versión anterior si v2 falla)

---

## 7. Ventajas Competitivas

### 7.1 MillenniumDB Native GNN vs Neo4j GDS

| Característica | Neo4j GDS | MillenniumDB (Propuesto) |
|----------------|-----------|--------------------------|
| **Arquitecturas GNN** | 2 fijas (GraphSAGE, HashGNN) | **5+ flexibles** (GCN, GAT, GraphSAGE, GIN, custom) |
| **Persistencia de modelos** | ❌ Volátil (se pierde al reiniciar) | ✅ **Persistente en disco** |
| **Versionado de modelos** | ❌ Manual (requiere MLflow externo) | ✅ **Automático** (v1, v2, v3...) |
| **Snapshots de datos** | ❌ No existe | ✅ **SHA256 hash** de proyección |
| **Reproducibilidad** | ⚠️ Difícil (proyección volátil) | ✅ **Garantizada** (modelo + snapshot) |
| **Consultas híbridas** | ❌ No (GDS separado de Cypher) | ✅ **Sí** (CALL inline en queries) |
| **Custom architectures** | ❌ Solo Java plugin (complejo) | ✅ **Sí** (LibTorch C++) |
| **GPU support** | ⚠️ Limitado (solo Java) | ✅ **Full** (LibTorch CUDA) |
| **Training in DBMS** | ✅ Sí (Java nativo) | ✅ Sí (C++ nativo) |
| **Inference in queries** | ❌ No (solo batch) | ✅ **Sí** (inline prediction) |
| **Model catalog** | ❌ No existe | ✅ **JSON catalog** con metadata |
| **Multi-proyección ML** | ⚠️ Una proyección a la vez | ✅ **Múltiples** (cada una con modelos) |
| **Tamaño de grafo** | ⚠️ RAM (10-100M nodos típico) | ✅ **Disco** (ilimitado) |

**Resumen**: MillenniumDB ofrece:
- **Mayor flexibilidad**: Arquitecturas custom, múltiples proyecciones
- **Mejor persistencia**: Modelos + snapshots versionados
- **Mejor integración**: Predicción inline en queries GQL
- **Mejor escalabilidad**: Sin límite de RAM

---

### 7.2 MillenniumDB Native GNN vs External Training (PyTorch)

| Característica | External PyTorch | MillenniumDB Native |
|----------------|------------------|---------------------|
| **Simplicidad** | ⚠️ Complejo (exportar → script → importar) | ✅ **Simple** (todo en GQL) |
| **Latencia** | ❌ Alta (I/O + network) | ✅ **Baja** (datos ya en DBMS) |
| **Versionado** | ❌ Manual (MLflow, DVC) | ✅ **Automático** |
| **Reproducibilidad** | ⚠️ Difícil (deps, seeds, datos) | ✅ **Garantizada** (snapshot) |
| **Producción** | ⚠️ Complejo (servir modelo aparte) | ✅ **Simple** (CALL en query) |
| **Sincronización datos-modelo** | ❌ Manual (re-exportar al cambiar datos) | ✅ **Automática** (snapshot check) |
| **Consultas híbridas** | ❌ Imposible | ✅ **Posible** |
| **Overhead** | ⚠️ Alto (exportar/importar) | ✅ **Bajo** (cero I/O extra) |
| **Flexibilidad** | ✅ Total (cualquier modelo PyTorch) | ⚠️ Limitada (arquitecturas soportadas) |
| **Debugging** | ✅ Fácil (Jupyter, TensorBoard) | ⚠️ Más difícil (dentro de DBMS) |

**Resumen**:
- **Native GNN** es mejor para: Producción, simplicidad, latencia, reproducibilidad
- **External PyTorch** es mejor para: Investigación, experimentación, debugging, custom loss functions

**Recomendación**: Usar ambos:
1. **Experimentación**: PyTorch externo (notebooks, TensorBoard)
2. **Producción**: MillenniumDB native (después de encontrar arquitectura óptima)

---

### 7.3 Ventaja Única: "Graph-Model Co-location"

**Concepto revolucionario**: Modelo ML vive **junto con** los datos en el DBMS

```
data/dbs/gql/mi_db/projections/social_network/
├── node_id.{dir,leaf}           ← DATOS
├── from_to_edge.{dir,leaf}      ← DATOS
├── node_key_value.{dir,leaf}    ← DATOS (features)
│
└── models/
    ├── user_classifier_v1.pt    ← MODELO ML
    ├── user_classifier_v2.pt    ← MODELO ML
    └── model_catalog.json       ← METADATA ML
```

**Beneficios**:

1. **Snapshot atómico**:
```bash
# Backup de datos + modelos en un solo comando
tar -czf backup_20251018.tar.gz data/dbs/gql/mi_db/projections/social_network/

# Restore recupera DATOS + MODELOS simultáneamente
tar -xzf backup_20251018.tar.gz
```

2. **Deployment simplificado**:
```bash
# Tradicional (PyTorch externo):
# 1. Copiar base de datos
# 2. Copiar modelos ML por separado
# 3. Configurar paths, sincronizar versiones
# 4. Verificar compatibilidad...

# MillenniumDB native:
cp -r projections/social_network/ /production/
# ¡Listo! Datos + modelos + metadata copiados atómicamente
```

3. **Versionado unificado**:
```json
{
  "model_id": "user_classifier_v3",
  "projection_snapshot": "sha256:abc123...",  ← Vincula modelo a datos exactos
  "created_at": "2025-10-18T10:30:00Z"
}
```

**Analogía**:
- **Tradicional**: Base de datos en `/var/lib/postgres/`, modelos ML en `/opt/models/` → desacoplados
- **MillenniumDB Native**: Todo en `/projections/social_network/` → acoplados, atómicos, versionados

---

### 7.4 Caso de Uso Único: "Time-Travel ML"

**Pregunta**: ¿Cómo se comportaba mi modelo hace 3 meses con los datos de ese momento?

**Con MillenniumDB Native**:

```gql
-- 1. Listar snapshots históricos
CALL mdb.gnn.list_models()
YIELD model_id, created_at, projection_snapshot, test_accuracy
ORDER BY created_at

-- Resultado:
-- model_id           | created_at          | snapshot      | test_acc
-- -------------------|---------------------|---------------|----------
-- classifier_v1      | 2025-07-15T10:00:00 | sha256:aaa... | 0.85
-- classifier_v2      | 2025-08-20T14:30:00 | sha256:bbb... | 0.88
-- classifier_v3      | 2025-10-18T10:30:00 | sha256:ccc... | 0.91

-- 2. Verificar si proyección actual coincide con snapshot de v1
USE "social_network"
CALL mdb.gnn.verify_snapshot("classifier_v1")
YIELD valid, current_snapshot, expected_snapshot

-- Resultado:
-- valid=false (proyección cambió desde julio)

-- 3. Re-crear proyección histórica (si tenemos backup)
-- (restaurar desde backup de julio)

-- 4. Evaluar modelo v1 con datos históricos
USE "social_network_july_backup"
CALL mdb.gnn.evaluate("classifier_v1")
YIELD test_accuracy

-- Resultado: test_acc = 0.85 (mismo que en julio)
-- ✅ Reproducibilidad perfecta!
```

**Aplicación**: Auditoría ML, debugging de degradación de modelos, compliance

---

## 8. Roadmap de Implementación

### 8.1 Visión General

**Objetivo**: Integrar entrenamiento de GNN nativo en MillenniumDB aprovechando LibTorch

**Fases**:
```
Fase 0: Infraestructura Base (Pre-requisito)    → 3-4 meses
Fase 1: Motor GNN Core                          → 4-6 meses
Fase 2: Procedimientos GQL                      → 2-3 meses
Fase 3: Optimizaciones + GPU                    → 2-3 meses
Fase 4: Arquitecturas Avanzadas                 → 3-4 meses
```

**Total**: 14-20 meses (1-2 developers)

---

### 8.2 Fase 0: Infraestructura Base (Pre-requisito)

**Duración**: 3-4 meses
**Equipo**: 1-2 developers
**Dependencias**: Ninguna

**Componentes**:

#### 0.1 Instalación de LibTorch

```cmake
# CMakeLists.txt
find_package(Torch REQUIRED)
target_link_libraries(mdb ${TORCH_LIBRARIES})
```

**Tareas**:
- [ ] Descargar LibTorch 2.1+ (CPU + CUDA versions)
- [ ] Integrar en CMake build system
- [ ] Verificar compatibilidad con GCC 8+
- [ ] Tests de integración básica (crear tensor, forward pass simple)

**Tiempo**: 1 semana

---

#### 0.2 Tipo VECTOR en ObjectId

**Archivo**: `src/graph_models/object_id.h`

**Tareas**:
- [ ] Añadir tipos `VECTOR_FLOAT_INLINE`, `VECTOR_FLOAT_EXTERN`
- [ ] Implementar serialización/deserialización
- [ ] Implementar VectorStore (similar a StringDictionary)
- [ ] Tests unitarios

**Código** (~300 líneas):
```cpp
// Nuevo tipo en ObjectId
enum class Type : uint8_t {
    VECTOR_FLOAT_INLINE  = 0x70,  // Up to 7 floats inline
    VECTOR_FLOAT_EXTERN  = 0x71,  // Reference to VectorStore
};

class VectorStore {
    // Similar a StringDictionary
    void insert(const std::vector<float>& vec);
    std::vector<float> get(uint64_t id);
};
```

**Tiempo**: 2 semanas

---

#### 0.3 MUTATE PROPERTY Syntax

**Archivo**: `src/query/parser/grammar/gql/GQL.g4`

**Tareas**:
- [ ] Añadir gramática para MUTATE PROPERTY
- [ ] Implementar visitor en query_visitor.cc
- [ ] Implementar binding iter para escritura de propiedades
- [ ] Tests de integración

**Sintaxis**:
```gql
MATCH (n)
WHERE n.age > 30
MUTATE PROPERTY predicted_class = 5,
                confidence = 0.95
```

**Tiempo**: 2-3 semanas

---

#### 0.4 Procedimientos GQL (Framework Base)

**Archivo**: `src/query/executor/binding_iter/procedures/procedure_registry.{h,cc}`

**Tareas**:
- [ ] Crear sistema de registro de procedimientos
- [ ] Implementar CALL syntax en parser
- [ ] Implementar binding iter base para procedimientos
- [ ] Ejemplo: `CALL mdb.util.echo("hello")`
- [ ] Tests

**Tiempo**: 2 semanas

---

**Total Fase 0**: 3-4 meses

---

### 8.3 Fase 1: Motor GNN Core

**Duración**: 4-6 meses
**Equipo**: 2 developers
**Dependencias**: Fase 0 completa

**Componentes**:

#### 1.1 Model Catalog

**Archivos**:
- `src/graph_models/gql/ml/model_catalog.{h,cc}` (~400 líneas)

**Tareas**:
- [ ] Implementar ModelMetadata struct
- [ ] Implementar persistencia JSON (load/save)
- [ ] Versionado automático
- [ ] Tests unitarios

**Tiempo**: 2 semanas

---

#### 1.2 GNN Model Wrappers

**Archivos**:
- `src/graph_models/gql/ml/gnn_model.{h,cc}` (~800 líneas)
- `src/graph_models/gql/ml/gcn_model.cc` (~300 líneas)
- `src/graph_models/gql/ml/gat_model.cc` (~400 líneas)
- `src/graph_models/gql/ml/graphsage_model.cc` (~400 líneas)

**Tareas**:
- [ ] GNNModel base class
- [ ] GCNModel implementation (message passing)
- [ ] GATModel implementation (attention)
- [ ] GraphSAGEModel implementation (sampling)
- [ ] GNNModelFactory
- [ ] Tests unitarios (forward pass con tensores pequeños)

**Tiempo**: 6-8 semanas

---

#### 1.3 Training Pipeline

**Archivos**:
- `src/graph_models/gql/ml/trainer.{h,cc}` (~600 líneas)

**Tareas**:
- [ ] GNNTrainer class
- [ ] Forward/backward pass loop
- [ ] Early stopping
- [ ] Checkpointing
- [ ] Progress callbacks
- [ ] Tests con grafos pequeños (10-100 nodos)

**Tiempo**: 4 semanas

---

#### 1.4 Graph Data Loader

**Archivos**:
- `src/graph_models/gql/ml/graph_data_loader.{h,cc}` (~800 líneas)

**Tareas**:
- [ ] Stream de B+Trees a PyTorch tensors
- [ ] Cargar features (node_key_value → tensor)
- [ ] Cargar topología (from_to_edge → edge_index)
- [ ] Cargar labels y masks
- [ ] Mapeo node_id → índice 0-based
- [ ] Tests con proyecciones reales

**Tiempo**: 4-6 semanas

---

#### 1.5 Projection Snapshots

**Archivos**:
- `src/graph_models/gql/projection/projection_snapshot.{h,cc}` (~200 líneas)

**Tareas**:
- [ ] Computar SHA256 hash de proyección
- [ ] Verificar snapshot
- [ ] Integrar con ModelCatalog
- [ ] Tests

**Tiempo**: 1 semana

---

**Total Fase 1**: 4-6 meses

---

### 8.4 Fase 2: Procedimientos GQL

**Duración**: 2-3 meses
**Equipo**: 1-2 developers
**Dependencias**: Fase 1 completa

**Archivos a crear**:
```
src/query/executor/binding_iter/procedures/gql/
├── procedure_gnn_define.{h,cc}           (~300 líneas)
├── procedure_gnn_configure_data.{h,cc}   (~200 líneas)
├── procedure_gnn_train.{h,cc}            (~500 líneas)
├── procedure_gnn_predict.{h,cc}          (~600 líneas)
├── procedure_gnn_evaluate.{h,cc}         (~300 líneas)
├── procedure_gnn_list_models.{h,cc}      (~200 líneas)
└── procedure_gnn_delete_model.{h,cc}     (~150 líneas)
```

**Total**: ~2,250 líneas C++

**Tareas**:

#### 2.1 CALL mdb.gnn.define()
- [ ] Implementar procedure
- [ ] Integrar con ModelCatalog
- [ ] Tests end-to-end

**Tiempo**: 1 semana

---

#### 2.2 CALL mdb.gnn.configure_data()
- [ ] Implementar procedure
- [ ] Validación de features/labels
- [ ] Tests

**Tiempo**: 1 semana

---

#### 2.3 CALL mdb.gnn.train()
- [ ] Implementar procedure
- [ ] Integrar con Trainer
- [ ] Yield de progreso (epoch, loss, acc)
- [ ] Tests con grafos pequeños (~1000 nodos)

**Tiempo**: 2-3 semanas

---

#### 2.4 CALL mdb.gnn.predict()
- [ ] Modo batch (todos los nodos)
- [ ] Modo inline (nodo individual en query)
- [ ] Integración con MUTATE PROPERTY
- [ ] Tests

**Tiempo**: 3-4 semanas

---

#### 2.5 Otros procedimientos
- [ ] mdb.gnn.evaluate()
- [ ] mdb.gnn.list_models()
- [ ] mdb.gnn.delete_model()

**Tiempo**: 2 semanas

---

**Total Fase 2**: 2-3 meses

---

### 8.5 Fase 3: Optimizaciones + GPU

**Duración**: 2-3 meses
**Equipo**: 1-2 developers
**Dependencias**: Fase 2 completa

**Componentes**:

#### 3.1 GPU Support

**Tareas**:
- [ ] Verificar instalación CUDA
- [ ] Mover tensores a GPU automáticamente
- [ ] Tests en GPU (si disponible)
- [ ] Fallback a CPU si no hay GPU

**Código**:
```cpp
torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
model->to(device);
data.x = data.x.to(device);
```

**Tiempo**: 2 semanas

---

#### 3.2 Mini-batch Training (para grafos grandes)

**Tareas**:
- [ ] Neighborhood sampling (k-hop)
- [ ] GraphSAINT sampling
- [ ] Batch data loader
- [ ] Tests con grafos >1M nodos

**Tiempo**: 4-6 semanas

---

#### 3.3 Model Caching

**Tareas**:
- [ ] Cache de modelos en RAM (evitar cargar .pt cada vez)
- [ ] LRU eviction
- [ ] Tests de performance

**Tiempo**: 2 semanas

---

#### 3.4 Paralelización

**Tareas**:
- [ ] Entrenamiento multi-thread (LibTorch)
- [ ] Inferencia batch paralela
- [ ] Tests de speedup

**Tiempo**: 2 semanas

---

**Total Fase 3**: 2-3 meses

---

### 8.6 Fase 4: Arquitecturas Avanzadas

**Duración**: 3-4 meses
**Equipo**: 1-2 developers
**Dependencias**: Fase 3 completa

**Componentes**:

#### 4.1 GIN (Graph Isomorphism Network)

**Tareas**:
- [ ] Implementar GINModel
- [ ] Tests
- [ ] Benchmark vs GCN/GAT

**Tiempo**: 3 semanas

---

#### 4.2 Edge-level GNNs (para link prediction)

**Tareas**:
- [ ] Implementar edge embeddings
- [ ] Scoring functions (dot product, MLP)
- [ ] CALL mdb.gnn.predict_link()
- [ ] Tests

**Tiempo**: 4 semanas

---

#### 4.3 Heterogeneous GNNs

**Tareas**:
- [ ] Soporte para múltiples tipos de nodos
- [ ] Soporte para múltiples tipos de aristas
- [ ] HeteroGraphSAGE
- [ ] Tests

**Tiempo**: 6 semanas

---

#### 4.4 Custom Architecture DSL (opcional)

**Idea**: Permitir definir arquitecturas custom en GQL

```gql
CALL mdb.gnn.define_custom("my_architecture", {
    layers: [
        {type: "GCN", hidden_dim: 128},
        {type: "GAT", hidden_dim: 64, num_heads: 8},
        {type: "GCN", hidden_dim: 32}
    ]
})
```

**Tiempo**: 6-8 semanas (si se decide implementar)

---

**Total Fase 4**: 3-4 meses

---

### 8.7 Timeline y Milestones

```
Mes  0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20
     ├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
Fase 0: [████████████]
           Infraestructura Base
           - LibTorch integration
           - VECTOR type
           - MUTATE PROPERTY
           - Procedure framework

Fase 1:           [████████████████████████]
                     Motor GNN Core
                     - Model catalog
                     - GNN implementations
                     - Training pipeline
                     - Data loader

Fase 2:                                    [████████████]
                                              Procedimientos GQL
                                              - CALL mdb.gnn.define
                                              - CALL mdb.gnn.train
                                              - CALL mdb.gnn.predict

Fase 3:                                                  [████████]
                                                            Optimizaciones
                                                            - GPU support
                                                            - Mini-batch
                                                            - Caching

Fase 4:                                                           [██████████]
                                                                     Avanzado
                                                                     - GIN
                                                                     - Link pred
                                                                     - Hetero
```

**Milestones**:

- **M1 (Mes 4)**: Fase 0 completa → LibTorch funcional, VECTOR type, MUTATE PROPERTY
- **M2 (Mes 10)**: Fase 1 completa → GCN/GAT/GraphSAGE entrenan en grafos pequeños
- **M3 (Mes 13)**: Fase 2 completa → **MVP funcional** (usuarios pueden entrenar modelos vía GQL)
- **M4 (Mes 16)**: Fase 3 completa → GPU support, escalabilidad a grafos grandes
- **M5 (Mes 20)**: Fase 4 completa → Arquitecturas avanzadas, producción-ready

---

### 8.8 Recursos Necesarios

**Equipo**:
- **2 developers senior** (C++, graph databases, ML background)
- **1 tech lead** (arquitectura, revisión de código)

**Hardware**:
- Servidor con GPU NVIDIA (para tests de GPU)
  - Recomendado: RTX 4090 o A100
- Servidor con disco SSD rápido (para proyecciones grandes)

**Software**:
- LibTorch 2.1+ (CPU + CUDA)
- CUDA 11.8+ (si se usa GPU)
- CMake 3.12+
- GCC 8+

**Datasets para testing**:
- Pequeño: Cora (2.7K nodos, 5.4K aristas)
- Mediano: PubMed (19K nodos, 44K aristas)
- Grande: ogbn-arxiv (169K nodos, 1.1M aristas)
- Muy grande: ogbn-papers100M (111M nodos, 1.6B aristas)

---

### 8.9 Riesgos y Mitigaciones

| Riesgo | Impacto | Probabilidad | Mitigación |
|--------|---------|--------------|------------|
| LibTorch incompatible con build system | Alto | Media | Probar integración en Fase 0 temprano |
| Performance de data loading (B+Tree → tensors) | Alto | Media | Optimizar con cacheo, paralelización |
| Memory overflow en grafos grandes | Alto | Alta | Implementar mini-batch sampling obligatorio |
| GPU CUDA compatibility issues | Medio | Media | Mantener soporte CPU como fallback |
| Complejidad de debugging (C++ + LibTorch) | Medio | Alta | Crear tests unitarios exhaustivos, logs detallados |
| Scope creep (demasiadas arquitecturas) | Medio | Media | Enfocarse en MVP (GCN, GAT, GraphSAGE) primero |

---

### 8.10 Alternativas Consideradas

#### Opción A: Usar ONNX Runtime (solo inferencia)

**Pros**:
- Más ligero que LibTorch
- Inferencia más rápida

**Contras**:
- ❌ **No permite entrenamiento** (objetivo principal)
- Requiere entrenar en PyTorch externo de todas formas

**Decisión**: Rechazada

---

#### Opción B: Implementar GNN desde cero (Eigen + autodiff manual)

**Pros**:
- Control total
- Sin dependencias grandes

**Contras**:
- ❌ Mucho trabajo (6-12 meses solo para autodiff)
- ❌ No GPU support fácil
- ❌ Reinventar la rueda

**Decisión**: Rechazada

---

#### Opción C: LibTorch (ELEGIDA)

**Pros**:
- ✅ Ecosistema completo (training + inference)
- ✅ GPU support nativo
- ✅ Compatible con PyTorch (modelos intercambiables)
- ✅ Mantenido activamente

**Contras**:
- Dependencia grande (~500MB)
- Curva de aprendizaje C++ API

**Decisión**: ✅ **ELEGIDA**

---

## 9. Conclusiones

### 9.1 Resumen de Propuesta

Este documento propone la integración de **entrenamiento de GNN nativo** en MillenniumDB, permitiendo:

1. **Entrenar modelos GNN directamente en el DBMS** (no necesita exportar datos)
2. **Persistir modelos junto con datos** (co-location)
3. **Versionado automático** de modelos y snapshots de datos
4. **Inferencia inline en queries GQL** (consultas híbridas graph + ML)
5. **Reproducibilidad garantizada** (modelo + datos versionados)

---

### 9.2 Diferenciación Competitiva

**vs Neo4j GDS**:
- ✅ Más arquitecturas GNN (5+ vs 2)
- ✅ Modelos persistentes (vs volátiles)
- ✅ Consultas híbridas (vs separadas)
- ✅ Escalabilidad ilimitada (disco vs RAM)

**vs PyTorch Externo**:
- ✅ Simplicidad (todo en GQL vs exportar/importar)
- ✅ Latencia baja (cero I/O extra)
- ✅ Reproducibilidad (snapshots automáticos)

---

### 9.3 Impacto Esperado

**Académico**:
- Paper en SIGMOD/VLDB: "Native GNN Training in Graph Databases"
- Benchmark vs Neo4j GDS

**Industrial**:
- Caso de uso killer: Fraud detection en tiempo real (graph query + ML prediction)
- Adoption en empresas que ya usan MillenniumDB

---

### 9.4 Próximos Pasos

1. **Validación con stakeholders** (equipo MillenniumDB)
2. **Aprobación de roadmap** (6-20 meses)
3. **Asignación de recursos** (2 developers)
4. **Inicio de Fase 0** (LibTorch integration)

---

**Fin del Documento**

---

**Autores**: Claude + Benito
**Fecha**: Octubre 18, 2025
**Versión**: 1.0
**Palabras**: ~15,000
**Líneas de código (propuestas)**: ~8,000 C++