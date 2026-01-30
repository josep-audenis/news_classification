import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets, model_selection, metrics, decomposition, neighbors
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
import warnings
import time
import os

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
sns.set_style('whitegrid')

if not os.path.exists('assets'):
    os.makedirs('assets')


digits = datasets.load_digits()
X_digits = digits.data
Y_digits = digits.target

print(f"\nDataset shape: X={X_digits.shape}, Y={Y_digits.shape}")
print(f"Total samples: {X_digits.shape[0]}")
print(f"Features per sample: {X_digits.shape[1]} (8x8 pixel images)")
print(f"Number of classes: {len(np.unique(Y_digits))}")

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle('Mostres de Dígits del Dataset', fontsize=14, fontweight='bold')
for i, ax in enumerate(axes.flat):
    ax.imshow(X_digits[i].reshape(8, 8), cmap='gray')
    ax.set_title(f'Etiqueta: {Y_digits[i]}')
    ax.axis('off')
plt.tight_layout()
plt.savefig('assets/digits_samples.png', dpi=150, bbox_inches='tight')

X_train_dig, X_test_dig, Y_train_dig, Y_test_dig = model_selection.train_test_split(
    X_digits, Y_digits, test_size=0.3, random_state=42, stratify=Y_digits
)

print(f"Train set: {X_train_dig.shape[0]} mostres ({X_train_dig.shape[0]/len(X_digits)*100:.1f}%)")
print(f"Test set: {X_test_dig.shape[0]} mostres ({X_test_dig.shape[0]/len(X_digits)*100:.1f}%)")

scaler_dig = StandardScaler()
X_train_dig_norm = scaler_dig.fit_transform(X_train_dig)
X_test_dig_norm = scaler_dig.transform(X_test_dig)


pca_dig = decomposition.PCA(n_components=2)
X_train_pca_dig = pca_dig.fit_transform(X_train_dig_norm)

svd_dig = decomposition.TruncatedSVD(n_components=2, random_state=42)
X_train_svd_dig = svd_dig.fit_transform(X_train_dig_norm)

lda_dig = LinearDiscriminantAnalysis(n_components=2)
X_train_lda_dig = lda_dig.fit_transform(X_train_dig_norm, Y_train_dig)

print(f"PCA Exp. Variance: {pca_dig.explained_variance_ratio_.sum()*100:.2f}%")
print(f"LDA Exp. Variance: {lda_dig.explained_variance_ratio_.sum()*100:.2f}%")

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

scatter1 = ax1.scatter(X_train_pca_dig[:, 0], X_train_pca_dig[:, 1], c=Y_train_dig, cmap='tab10', alpha=0.6, edgecolors='k', linewidth=0.5)
ax1.set_title('PCA Projection')
ax1.grid(True, alpha=0.3)
plt.colorbar(scatter1, ax=ax1, label='Dígit')

scatter2 = ax2.scatter(X_train_svd_dig[:, 0], X_train_svd_dig[:, 1], c=Y_train_dig, cmap='tab10', alpha=0.6, edgecolors='k', linewidth=0.5)
ax2.set_title('Truncated SVD Projection')
ax2.grid(True, alpha=0.3)
plt.colorbar(scatter2, ax=ax2, label='Dígit')

scatter3 = ax3.scatter(X_train_lda_dig[:, 0], X_train_lda_dig[:, 1], c=Y_train_dig, cmap='tab10', alpha=0.6, edgecolors='k', linewidth=0.5)
ax3.set_title('LDA Projection')
ax3.grid(True, alpha=0.3)
plt.colorbar(scatter3, ax=ax3, label='Dígit')

plt.tight_layout()
plt.savefig('assets/digits_dimensionality_reduction.png', dpi=150, bbox_inches='tight')

def compute_test(X_train, Y_train, clf, cv=10):
    scores = model_selection.cross_val_score(clf, X_train, Y_train, cv=cv, scoring='accuracy')
    return scores

k_values = range(1, 21)
mean_scores = []
std_scores = []

for k in k_values:
    knn = neighbors.KNeighborsClassifier(n_neighbors=k)
    scores = compute_test(X_train_dig_norm, Y_train_dig, knn, cv=10)
    mean_scores.append(np.mean(scores))
    std_scores.append(np.std(scores))

optimal_k = k_values[np.argmax(mean_scores)]
print(f"\nValor optim de K: {optimal_k} amb accuracy: {max(mean_scores):.4f}")

fig, ax = plt.subplots(figsize=(12, 6))
ax.errorbar(k_values, mean_scores, yerr=std_scores, marker='o', capsize=5, capthick=2, linewidth=2, color='steelblue')
ax.axvline(x=optimal_k, color='r', linestyle='--', linewidth=2, label=f'K òptim = {optimal_k}')
ax.set_title('Performance de K-NN vs Nombre de Veïns (CV=10)')
ax.legend()
plt.savefig('assets/digits_k_optimization.png', dpi=150, bbox_inches='tight')



param_grid = {
    'n_neighbors': [1, 3, 5, 7, optimal_k, 11],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan']
}

scenarios_dig = {
    'Original (64 dims)': X_train_dig_norm,
    'PCA (30 dims)': decomposition.PCA(n_components=30).fit_transform(X_train_dig_norm),
    'PCA (20 dims)': decomposition.PCA(n_components=20).fit_transform(X_train_dig_norm),
    'LDA (9 dims)': LinearDiscriminantAnalysis(n_components=9).fit_transform(X_train_dig_norm, Y_train_dig)
}

best_dig_score = 0
best_dig_params = None
best_dig_scenario = None
best_dig_model = None
results_dig = []

transformers = {
    'PCA (30 dims)': decomposition.PCA(n_components=30).fit(X_train_dig_norm),
    'PCA (20 dims)': decomposition.PCA(n_components=20).fit(X_train_dig_norm),
    'LDA (9 dims)': LinearDiscriminantAnalysis(n_components=9).fit(X_train_dig_norm, Y_train_dig)
}

for scenario_name, X_scenario in scenarios_dig.items():
    print(f" - Scenario: {scenario_name}")
    knn = neighbors.KNeighborsClassifier()
    grid_search = model_selection.GridSearchCV(knn, param_grid, cv=10, scoring='accuracy', n_jobs=-1, verbose=0)
    grid_search.fit(X_scenario, Y_train_dig)
    
    results_dig.append({
        'scenario': scenario_name,
        'best_score': grid_search.best_score_,
        'best_params': grid_search.best_params_
    })
    
    if grid_search.best_score_ > best_dig_score:
        best_dig_score = grid_search.best_score_
        best_dig_params = grid_search.best_params_
        best_dig_scenario = scenario_name
        best_dig_model = grid_search.best_estimator_

print(f"MILLOR CONFIGURACIÓ KNN:")
print(f" - Scenario: {best_dig_scenario}")
print(f" - Parameters: {best_dig_params}")
print(f" - CV Accuracy: {best_dig_score:.4f}")

fig, ax = plt.subplots(figsize=(10, 5))
scenario_names = [r['scenario'] for r in results_dig]
scenario_scores = [r['best_score'] for r in results_dig]
bars = ax.bar(range(len(scenario_names)), scenario_scores, color='steelblue', edgecolor='black')
ax.set_xticks(range(len(scenario_names)))
ax.set_xticklabels(scenario_names, rotation=15, ha='right')
ax.set_ylim([0.90, 1.0])
ax.set_title('KNN: Comparació de Performance segons Dimensionalitat')
for i, bar in enumerate(bars):
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{scenario_scores[i]:.4f}',
            ha='center', va='bottom', fontweight='bold')
plt.tight_layout()
plt.savefig('assets/digits_scenario_comparison.png', dpi=150, bbox_inches='tight')

if best_dig_scenario in transformers:
    X_test_final_dig = transformers[best_dig_scenario].transform(X_test_dig_norm)
else:
    X_test_final_dig = X_test_dig_norm

Y_pred_dig = best_dig_model.predict(X_test_final_dig)
test_acc_dig = metrics.accuracy_score(Y_test_dig, Y_pred_dig)

print(f"Test Set Accuracy (Best KNN): {test_acc_dig:.4f}")
print(metrics.classification_report(Y_test_dig, Y_pred_dig))


learners_digits = {
    'KNN (Optimized)': best_dig_model, 
    'MLP (Neural Net)': MLPClassifier(hidden_layer_sizes=(100, 100), learning_rate_init=0.02, max_iter=300, random_state=42),
    'AdaBoost': AdaBoostClassifier(n_estimators=100, random_state=42), 
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42)
}

results_comparison_digits = {}

print(f"{'Model':<20} | {'Test Accuracy':<15}")
print("-" * 40)

for name, model in learners_digits.items():
    if name == 'KNN (Optimized)':
        acc = test_acc_dig
    else:
        model.fit(X_train_dig_norm, Y_train_dig)
        acc = metrics.accuracy_score(Y_test_dig, model.predict(X_test_dig_norm))
    
    results_comparison_digits[name] = acc
    print(f"{name:<20} | {acc:.4f}")

plt.figure(figsize=(10, 5))
plt.bar(results_comparison_digits.keys(), results_comparison_digits.values(), color=['skyblue', 'orange', 'green', 'purple'])
plt.ylim(0.8, 1.0)
plt.title('Comparativa Learners sobre DÍGITS (Test Set)')
plt.savefig('assets/digits_learners_comparison.png')


categories = [
    'comp.graphics',
    'comp.sys.mac.hardware',
    'rec.sport.baseball',
    'sci.med',
    'sci.space',
    'talk.politics.misc'
]

newsgroups_train = datasets.fetch_20newsgroups(
    subset='train', categories=categories, shuffle=True, random_state=42,
    remove=('headers', 'footers', 'quotes')
)
newsgroups_test = datasets.fetch_20newsgroups(
    subset='test', categories=categories, shuffle=True, random_state=42,
    remove=('headers', 'footers', 'quotes')
)

print(f"Train: {len(newsgroups_train.data)} documents")
print(f"Test: {len(newsgroups_test.data)} documents")

unique_train, counts_train = np.unique(newsgroups_train.target, return_counts=True)
fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(range(len(counts_train)), counts_train, color='steelblue', edgecolor='black')
ax.set_yticks(range(len(counts_train)))
ax.set_yticklabels([newsgroups_train.target_names[i] for i in unique_train])
ax.set_title('Distribució de Categories - 20 Newsgroups')
plt.tight_layout()
plt.savefig('assets/newsgroups_distribution.png', dpi=150, bbox_inches='tight')



vectorizer = TfidfVectorizer(
    max_features=3000, 
    max_df=0.5,
    min_df=5,
    stop_words='english'
)

X_train_news = vectorizer.fit_transform(newsgroups_train.data).toarray()
X_test_news = vectorizer.transform(newsgroups_test.data).toarray()
Y_train_news = newsgroups_train.target
Y_test_news = newsgroups_test.target

print(f"Shape TF-IDF Train: {X_train_news.shape}")

svd_news = decomposition.TruncatedSVD(n_components=100, random_state=42)
X_train_news_reduced = svd_news.fit_transform(X_train_news)
X_test_news_reduced = svd_news.transform(X_test_news)
print(f"SVD Reduced Shape: {X_train_news_reduced.shape} (Var: {svd_news.explained_variance_ratio_.sum():.2f})")


results_news = {}

models_config = {
    'MLP (Neural Net)': {
        'model': MLPClassifier(max_iter=100, random_state=42, early_stopping=True),
        'params': {
            'hidden_layer_sizes': [(100,), (100, 50)],
            'learning_rate_init': [0.001, 0.01, 0.02], 
            'alpha': [0.0001, 0.001]
        }
    },
    'AdaBoost': {
        'model': AdaBoostClassifier(random_state=42), 
        'params': {
            'n_estimators': [50, 100],
            'learning_rate': [0.5, 1.0]
        }
    },
    'RandomForest': {
        'model': RandomForestClassifier(random_state=42, n_jobs=-1),
        'params': {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None]
        }
    },
    'KNN (Baseline)': {
        'model': neighbors.KNeighborsClassifier(),
        'params': {
            'n_neighbors': [5, 10, 20],
            'metric': ['cosine', 'euclidean'] 
        }
    }
}

for name, config in models_config.items():
    print(f"\nEvaluating {name}...")
    
    cv_folds = 5
    
    grid = model_selection.GridSearchCV(
        config['model'],
        config['params'],
        cv=cv_folds,
        n_jobs=-1,
        verbose=1
    )
    
    start_time = time.time()
    grid.fit(X_train_news_reduced, Y_train_news)
    elapsed = time.time() - start_time
    
    test_score = metrics.accuracy_score(Y_test_news, grid.predict(X_test_news_reduced))
    
    results_news[name] = {
        'cv_score': grid.best_score_,
        'test_score': test_score,
        'best_params': grid.best_params_,
        'time': elapsed,
        'model': grid.best_estimator_
    }
    
    print(f" - Best CV: {grid.best_score_:.4f}")
    print(f" - Test Acc: {test_score:.4f}")
    print(f" - Time: {elapsed:.2f}s")

print("-" * 55)

print(f"{'Model':<20} {'CV Score':>10} {'Test Acc':>10} {'Time(s)':>10}")
print("-" * 55)
for name, res in results_news.items():
    print(f"{name:<20} {res['cv_score']:>10.4f} {res['test_score']:>10.4f} {res['time']:>10.1f}")

best_text_model = max(results_news, key=lambda x: results_news[x]['test_score'])

fig, ax = plt.subplots(figsize=(10, 6))
models = list(results_news.keys())
scores = [results_news[m]['test_score'] for m in models]
colors = ['gold' if m == best_text_model else 'steelblue' for m in models]

bars = ax.bar(models, scores, color=colors, edgecolor='black')
ax.set_ylabel('Test Accuracy')
ax.set_title('Comparació Models - 20 Newsgroups')
ax.set_ylim(0, max(scores)*1.1)

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.3f}',
            ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('assets/newsgroups_model_comparison.png', dpi=150, bbox_inches='tight')

top_models = sorted(results_news.items(), key=lambda x: x[1]['test_score'], reverse=True)[:3]
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (name, res) in enumerate(top_models):
    y_pred = res['model'].predict(X_test_news_reduced)
    cm = metrics.confusion_matrix(Y_test_news, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', ax=axes[idx], cmap='Blues', cbar=False)
    axes[idx].set_title(f"{name}\nAcc: {res['test_score']:.3f}")
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('True')

plt.tight_layout()
plt.savefig('assets/newsgroups_confusion_matrices.png', dpi=150, bbox_inches='tight')
