# Manutenção Preditiva - Projeto Final Bootcamp CDIA

## Estrutura do Projeto
- **main.py**: script principal contendo pré-processamento, modelagem hierárquica (binário + multiclasse), treinamento, inferência e exportação de métricas/resultados.
- **bootcamp_train.csv**: dados de treino rotulados.
- **bootcamp_test.csv**: dados de teste sem rótulos, utilizados apenas para inferência.
- **metrics_algos_hierarquico_calibrado.json**: métricas geradas no treinamento.
- **pred_test_calibrado_rf.csv**: predições em teste para Random Forest (versão calibrada).

## Pipeline Implementado
1. **Pré-processamento**
   - Limpeza e imputação de dados.
   - Codificação one-hot da variável categórica `tipo`.
   - Criação de features derivadas: `delta_temp`, `potencia_mecanica`.

2. **Modelagem Hierárquica**
   - **Estágio A**: classificador binário (falha vs. sem falha).
     - Random Forest com `class_weight={0:1,1:8}` e calibração de probabilidades.
     - Threshold ótimo calculado via métrica F2.
   - **Estágio B**: classificador multiclasse (tipos de falha).
     - Random Forest com `class_weight="balanced"`.
     - Oversampling via SMOTE direcionado (reforço maior em falha aleatória e desgaste).
     - Calibração de probabilidades aplicada.

3. **Inferência**
   - Combinação hierárquica: P(tipo) = P(falha) * P(tipo | falha).
   - Normalização das probabilidades com guardrails numéricos.
   - Exportação em CSV das probabilidades e da classe prevista.

4. **Métricas**
   - Relatórios de classificação (precision, recall, f1-score) para ambos os estágios.
   - Matriz de confusão binária e multiclasse.
   - Threshold ótimo armazenado no modelo.

## Dependências
- Python 3.10+
- scikit-learn
- imbalanced-learn
- pandas
- numpy
- joblib

Instalação rápida:
```bash
pip install -r requirements.txt
```

## Execução
Rodar o pipeline completo:
```bash
python main.py
```

Saídas principais:
- `metrics_algos_hierarquico_calibrado.json`
- `pred_test_calibrado_rf.csv`
- `modelo_hierarquico_rf_calibrado.pkl`

## Observações
- Dados de treino e teste devem estar no diretório raiz do projeto.
- O pipeline está preparado para Random Forest, Decision Tree e SVM, mas o RF apresentou os melhores resultados.

## Ambiente Virtual

Recomendado criar um ambiente isolado para instalar as dependências:

```bash
python3 -m venv env
source env/bin/activate   # Linux/Mac
env\Scripts\activate    # Windows

pip install -r requirements.txt
```
