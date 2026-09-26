from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
	accuracy_score,
	classification_report,
	confusion_matrix,
	f1_score,
	precision_score,
	recall_score,
	roc_auc_score,
	roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "cc.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"
RANDOM_STATE = 43


def build_models() -> dict[str, object]:
	return {
		"Logistic regression": make_pipeline(
			StandardScaler(), LogisticRegression(max_iter=3000)
		)
	}


def build_mlp_search(cross_validator: StratifiedKFold) -> GridSearchCV:
	pipeline = make_pipeline(
		StandardScaler(),
		MLPClassifier(
			activation="relu",
			solver="adam",
			batch_size=32,
			learning_rate_init=0.001,
			max_iter=1800,
			n_iter_no_change=35,
			random_state=RANDOM_STATE,
		),
	)
	return GridSearchCV(
		pipeline,
		{
			"mlpclassifier__hidden_layer_sizes": [
				(32, 16),
				(64, 32),
				(64, 32, 16),
				(128, 64),
				(128, 64, 32),
			],
			"mlpclassifier__alpha": [0.0001, 0.001],
			"mlpclassifier__early_stopping": [True, False],
		},
		scoring={
			"accuracy": "accuracy",
			"precision": "precision",
			"recall": "recall",
			"f1": "f1",
			"roc_auc": "roc_auc",
		},
		refit="accuracy",
		cv=cross_validator,
		n_jobs=1,
	)


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
	if not DATA_PATH.exists():
		raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

	data = pd.read_csv(DATA_PATH)
	if "diagnosis" not in data.columns:
		raise ValueError("The dataset must contain a 'diagnosis' column.")

	labels = data["diagnosis"].map({"B": 0, "M": 1})
	if labels.isna().any():
		raise ValueError("Diagnosis values must be 'B' (benign) or 'M' (malignant).")

	features = data.drop(columns=["diagnosis", "id", "Unnamed: 32"], errors="ignore")
	features = features.select_dtypes(include=[np.number])
	if features.empty or features.isna().any().any():
		raise ValueError("The dataset must contain numeric features without missing values.")

	return features, labels.astype(int)


def save_evaluation_chart(
	models: dict[str, object],
	x_test: pd.DataFrame,
	y_test: pd.Series,
	selected_predictions: np.ndarray,
	selected_name: str,
) -> None:
	figure, (roc_axis, matrix_axis) = plt.subplots(1, 2, figsize=(12, 5))

	for name, model in models.items():
		probabilities = model.predict_proba(x_test)[:, 1]
		false_positive_rate, true_positive_rate, _ = roc_curve(y_test, probabilities)
		auc = roc_auc_score(y_test, probabilities)
		roc_axis.plot(false_positive_rate, true_positive_rate, label=f"{name} (AUC={auc:.3f})")

	roc_axis.plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=1)
	roc_axis.set(title="ROC curve", xlabel="False positive rate", ylabel="True positive rate")
	roc_axis.legend(loc="lower right", frameon=False)
	roc_axis.grid(alpha=0.2)

	matrix = confusion_matrix(y_test, selected_predictions, labels=[0, 1])
	matrix_axis.imshow(matrix, cmap="Blues")
	matrix_axis.set(
		title=f"{selected_name} confusion matrix",
		xlabel="Predicted diagnosis",
		ylabel="Actual diagnosis",
		xticks=[0, 1],
		yticks=[0, 1],
		xticklabels=["Benign", "Malignant"],
		yticklabels=["Benign", "Malignant"],
	)
	for row in range(2):
		for column in range(2):
			matrix_axis.text(column, row, str(matrix[row, column]), ha="center", va="center")

	figure.tight_layout()
	figure.savefig(OUTPUT_DIR / "model_evaluation.png", dpi=180, bbox_inches="tight")
	plt.close(figure)


def save_network_diagram(model: MLPClassifier, feature_count: int) -> None:
	layer_sizes = [feature_count, *model.hidden_layer_sizes, model.n_outputs_]
	layer_names = ["Input features"] + [
		f"Hidden layer {index}" for index in range(1, len(layer_sizes) - 1)
	] + ["Output"]
	x_positions = np.arange(len(layer_sizes)) * 2.1
	y_positions = [np.linspace(-1, 1, size) if size > 1 else np.array([0.0]) for size in layer_sizes]

	figure, axis = plt.subplots(figsize=(14, 9))
	for layer_index, weights in enumerate(model.coefs_):
		source_y, target_y = y_positions[layer_index], y_positions[layer_index + 1]
		for source_index, source_position in enumerate(source_y):
			for target_index, target_position in enumerate(target_y):
				weight = weights[source_index, target_index]
				color = "#d97706" if weight >= 0 else "#2563a6"
				axis.plot(
					[x_positions[layer_index], x_positions[layer_index + 1]],
					[source_position, target_position],
					color=color,
					alpha=0.11,
					linewidth=0.45,
					zorder=1,
				)

	for layer_index, positions in enumerate(y_positions):
		axis.scatter(
			np.full(len(positions), x_positions[layer_index]),
			positions,
			s=22,
			color="#f8fafc" if layer_index in (0, len(layer_sizes) - 1) else "#128277",
			edgecolors="#183b56",
			linewidths=0.7,
			zorder=2,
		)
		label = f"{layer_names[layer_index]}\n{layer_sizes[layer_index]} neurons"
		if layer_index == len(layer_sizes) - 1:
			label = f"{layer_names[layer_index]}\nmalignant probability"
		axis.text(x_positions[layer_index], -1.23, label, ha="center", va="top", fontsize=10)

	axis.set_title("Trained neural network architecture", fontsize=16, pad=18)
	axis.text(
		0.5,
		1.015,
		"Each line is a learned connection; orange and blue indicate positive and negative weights.",
		transform=axis.transAxes,
		ha="center",
		color="#475569",
	)
	axis.set_xlim(x_positions[0] - 0.7, x_positions[-1] + 0.7)
	axis.set_ylim(-1.65, 1.15)
	axis.axis("off")
	figure.tight_layout()
	figure.savefig(OUTPUT_DIR / "neural_network.png", dpi=180, bbox_inches="tight")
	plt.close(figure)


def main() -> None:
	OUTPUT_DIR.mkdir(exist_ok=True)
	features, labels = load_dataset()
	x_train, x_test, y_train, y_test = train_test_split(
		features,
		labels,
		test_size=0.2,
		random_state=RANDOM_STATE,
		stratify=labels,
	)
	cross_validator = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
	scoring = {
		"accuracy": "accuracy",
		"precision": "precision",
		"recall": "recall",
		"f1": "f1",
		"roc_auc": "roc_auc",
	}
	models = build_models()

	print(f"Dataset: {len(features)} samples, {features.shape[1]} features")
	print("5-fold nested cross-validation on training data:")
	logistic_model = models["Logistic regression"]
	logistic_scores = cross_validate(
		logistic_model, x_train, y_train, cv=cross_validator, scoring=scoring
	)
	logistic_model.fit(x_train, y_train)

	mlp_nested_scores = cross_validate(
		build_mlp_search(cross_validator),
		x_train,
		y_train,
		cv=cross_validator,
		scoring=scoring,
	)
	mlp_search = build_mlp_search(cross_validator)
	mlp_search.fit(x_train, y_train)
	mlp_model = mlp_search.best_estimator_
	print(f"  Selected MLP architecture: {mlp_search.best_params_['mlpclassifier__hidden_layer_sizes']}")

	models["Deep neural network"] = mlp_model
	cross_validation_scores = {
		"Logistic regression": logistic_scores,
		"Deep neural network": mlp_nested_scores,
	}
	cross_validation_accuracy = {
		name: scores["test_accuracy"].mean()
		for name, scores in cross_validation_scores.items()
	}
	for name, scores in cross_validation_scores.items():
		summary = " | ".join(
			f"{metric}: {scores[f'test_{metric}'].mean():.3f}"
			f" +/- {scores[f'test_{metric}'].std():.3f}"
			for metric in scoring
		)
		print(f"  {name}: {summary}")

	selected_name = max(cross_validation_accuracy, key=cross_validation_accuracy.get)
	selected_model = models[selected_name]
	predictions_by_model = {}
	metrics_rows = []
	for name, model in models.items():
		predictions = model.predict(x_test)
		probabilities = model.predict_proba(x_test)[:, 1]
		predictions_by_model[name] = predictions
		cv_scores = cross_validation_scores[name]
		metrics_rows.append(
			{
				"model": name,
				"cv_accuracy_mean": cv_scores["test_accuracy"].mean(),
				"cv_accuracy_std": cv_scores["test_accuracy"].std(),
				"cv_recall_mean": cv_scores["test_recall"].mean(),
				"cv_roc_auc_mean": cv_scores["test_roc_auc"].mean(),
				"test_accuracy": accuracy_score(y_test, predictions),
				"test_precision": precision_score(y_test, predictions),
				"test_recall": recall_score(y_test, predictions),
				"test_f1": f1_score(y_test, predictions),
				"test_roc_auc": roc_auc_score(y_test, probabilities),
			}
		)

	print(f"\nSelected by training cross-validation: {selected_name}")
	print("Held-out test results (both models):")
	for row in metrics_rows:
		print(
			f"  {row['model']}: accuracy={row['test_accuracy']:.3f}, "
			f"precision={row['test_precision']:.3f}, recall={row['test_recall']:.3f}, "
			f"F1={row['test_f1']:.3f}, ROC-AUC={row['test_roc_auc']:.3f}"
		)
	predictions = predictions_by_model[selected_name]
	print("\nClassification report:")
	print(classification_report(y_test, predictions, target_names=["Benign", "Malignant"]))

	mlp = mlp_model.named_steps["mlpclassifier"]
	save_evaluation_chart(models, x_test, y_test, predictions, selected_name)
	save_network_diagram(mlp, features.shape[1])
	pd.DataFrame(metrics_rows).to_csv(
		OUTPUT_DIR / "model_metrics.csv", index=False, float_format="%.4f"
	)
	joblib.dump(selected_model, OUTPUT_DIR / "breast_cancer_model.joblib")
	joblib.dump(mlp_model, OUTPUT_DIR / "breast_cancer_mlp.joblib")
	print("Generated outputs/model_evaluation.png")
	print("Generated outputs/neural_network.png")
	print("Generated outputs/model_metrics.csv")
	print(f"Saved outputs/breast_cancer_model.joblib ({selected_name})")
	print("Saved outputs/breast_cancer_mlp.joblib")


if __name__ == "__main__":
	main()
