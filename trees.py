import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score
)

from sklearn.tree import (
    DecisionTreeClassifier,
    plot_tree
)

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    precision_recall_fscore_support
)

from sklearn.datasets import (
    load_iris,
    load_breast_cancer
)

from scipy.stats import randint

# ЗАГРУЗКА ДАННЫХ


iris = load_iris()

X = iris.data
y = iris.target

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42
)


# ОБЫЧНОЕ ДЕРЕВО РЕШЕНИЙ


dt_classifier = DecisionTreeClassifier(random_state=42)

# Обучение модели
dt_classifier.fit(X_train, y_train)

# Предсказания
y_pred = dt_classifier.predict(X_test)

# Точность
accuracy = accuracy_score(y_test, y_pred)

print(f"Точность модели: {accuracy:.4f}")

# Подробный отчет
print(classification_report(
    y_test,
    y_pred,
    target_names=iris.target_names
))


# ДЕРЕВО С ОГРАНИЧЕНИЯМИ


dt_classifier_pruned = DecisionTreeClassifier(
    max_depth=3,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)

dt_classifier_pruned.fit(X_train, y_train)

y_pred_pruned = dt_classifier_pruned.predict(X_test)

accuracy_pruned = accuracy_score(y_test, y_pred_pruned)

print(f"Точность модели с ограниченной глубиной: {accuracy_pruned:.4f}")

# ВЕРОЯТНОСТИ КЛАССОВ


probabilities = dt_classifier.predict_proba(X_test)

print("Вероятности для первых 5 тестовых образцов:")

for i in range(5):
    print(f"Образец {i + 1}: {probabilities[i]}")


# ВАЖНОСТЬ ПРИЗНАКОВ


feature_importance = pd.DataFrame({
    'Feature': iris.feature_names,
    'Importance': dt_classifier.feature_importances_
}).sort_values(by='Importance', ascending=False)

print(feature_importance)


# GRID SEARCH


param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

dt = DecisionTreeClassifier(random_state=42)

grid_search = GridSearchCV(
    estimator=dt,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

print(f"Лучшие параметры: {grid_search.best_params_}")
print(f"Лучшая точность: {grid_search.best_score_:.4f}")

best_model = grid_search.best_estimator_

y_pred_best = best_model.predict(X_test)

accuracy_best = accuracy_score(y_test, y_pred_best)

print(f"Точность оптимизированной модели: {accuracy_best:.4f}")


# RANDOMIZED SEARCH


param_distributions = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None] + list(randint(3, 20).rvs(10)),
    'min_samples_split': randint(2, 20).rvs(10),
    'min_samples_leaf': randint(1, 10).rvs(10),
    'max_features': ['sqrt', 'log2', None]
}

random_search = RandomizedSearchCV(
    estimator=dt,
    param_distributions=param_distributions,
    n_iter=20,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42,
    verbose=1
)

random_search.fit(X_train, y_train)

print(f"Лучшие параметры Random Search: {random_search.best_params_}")
print(f"Лучшая точность: {random_search.best_score_:.4f}")


# COST COMPLEXITY PRUNING


dt_full = DecisionTreeClassifier(random_state=42)

path = dt_full.cost_complexity_pruning_path(X_train, y_train)

ccp_alphas = path.ccp_alphas
impurities = path.impurities

dt_models = []

for ccp_alpha in ccp_alphas:
    dt = DecisionTreeClassifier(
        random_state=42,
        ccp_alpha=ccp_alpha
    )

    dt.fit(X_train, y_train)

    dt_models.append(dt)

train_scores = [
    dt.score(X_train, y_train)
    for dt in dt_models
]

test_scores = [
    dt.score(X_test, y_test)
    for dt in dt_models
]

optimal_idx = np.argmax(test_scores)

optimal_alpha = ccp_alphas[optimal_idx]

print(f"Оптимальное значение ccp_alpha: {optimal_alpha:.6f}")

print(f"Точность на тестовых данных: {test_scores[optimal_idx]:.4f}")


# ВИЗУАЛИЗАЦИЯ ДЕРЕВА


dt_visual = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

dt_visual.fit(X_train, y_train)

plt.figure(figsize=(15, 10))

plot_tree(
    dt_visual,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True,
    rounded=True
)

plt.title("Дерево решений для набора данных Iris")

plt.show()


# ВАЖНОСТЬ ПРИЗНАКОВ ГРАФИК


importances = dt_visual.feature_importances_

feature_names = iris.feature_names

indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))

plt.title("Важность признаков")

plt.bar(
    range(len(indices)),
    importances[indices],
    align='center'
)

plt.xticks(
    range(len(indices)),
    [feature_names[i] for i in indices],
    rotation=90
)

plt.tight_layout()

plt.show()


# ОТСЛЕЖИВАНИЕ ПУТИ

def trace_path(tree, feature_names, sample):

    feature = tree.feature
    threshold = tree.threshold

    sample_array = np.array(
        sample,
        dtype=np.float32
    ).reshape(1, -1)

    node_indicator = tree.decision_path(
        sample_array
    ).toarray()[0]

    leaf_id = tree.apply(sample_array)[0]

    path = []

    for node_id in range(len(node_indicator)):

        if node_indicator[node_id] == 1:

            if leaf_id == node_id:

                class_probs = tree.value[node_id][0]

                predicted_class = np.argmax(class_probs)

                path.append(
                    f"Лист {node_id}: "
                    f"Предсказание = класс {predicted_class} "
                    f"(вероятности: {class_probs})"
                )

            else:

                feature_name = feature_names[
                    feature[node_id]
                ]

                feature_value = sample[
                    feature[node_id]
                ]

                if feature_value <= threshold[node_id]:

                    path.append(
                        f"Узел {node_id}: "
                        f"{feature_name} <= "
                        f"{threshold[node_id]:.2f} -> Да "
                        f"(значение: {feature_value:.2f})"
                    )

                else:

                    path.append(
                        f"Узел {node_id}: "
                        f"{feature_name} <= "
                        f"{threshold[node_id]:.2f} -> Нет "
                        f"(значение: {feature_value:.2f})"
                    )

    return path

sample_idx = 10

sample = X_test[sample_idx]

tree = dt_visual.tree_

path = trace_path(
    tree,
    iris.feature_names,
    sample
)

print(f"Путь для примера {sample_idx}:")

for step in path:
    print(step)

print(
    f"Фактический класс: "
    f"{iris.target_names[y_test[sample_idx]]}"
)

print(
    f"Предсказанный класс: "
    f"{iris.target_names[dt_visual.predict([sample])[0]]}"
)