import argparse
import copy
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
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
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "cc.csv"
ORIGINAL_DATA_PATH = PROJECT_DIR / "data" / "wisconsin_original.data"
OUTPUT_DIR = PROJECT_DIR / "outputs"
RANDOM_STATE = 43
SCORING = {
	"accuracy": "accuracy",
	"precision": "precision",
	"recall": "recall",
	"f1": "f1",
	"roc_auc": "roc_auc",
}


class TorchBreastCancerNet(nn.Module):
	def __init__(self, input_size: int) -> None:
		super().__init__()
		self.layers = nn.Sequential(
			nn.Linear(input_size, 64),
			nn.ReLU(),
			nn.Dropout(0.25),
			nn.Linear(64, 32),
			nn.ReLU(),
			nn.Dropout(0.15),
			nn.Linear(32, 1),
		)

	def forward(self, features: torch.Tensor) -> torch.Tensor:
		return self.layers(features).squeeze(1)


def resolve_device(requested: str) -> torch.device:
	if requested == "cuda" and not torch.cuda.is_available():
		raise RuntimeError("CUDA was requested, but PyTorch cannot access a CUDA device.")
	if requested == "cpu":
		return torch.device("cpu")
	return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fit_torch_model(
	features: pd.DataFrame,
	labels: pd.Series,
	device: torch.device,
	seed: int,
) -> dict[str, object]:
	x_fit, x_validation, y_fit, y_validation = train_test_split(
		features,
		labels,
		test_size=0.2,
		random_state=seed,
		stratify=labels,
	)
	imputer = SimpleImputer(strategy="median")
	scaler = StandardScaler()
	x_fit = scaler.fit_transform(imputer.fit_transform(x_fit)).astype(np.float32)
	x_validation = scaler.transform(imputer.transform(x_validation)).astype(np.float32)
	y_fit_array = y_fit.to_numpy(dtype=np.float32)
	y_validation_array = y_validation.to_numpy(dtype=np.float32)

	torch.manual_seed(seed)
	if device.type == "cuda":
		torch.cuda.manual_seed_all(seed)
		generator = torch.Generator(device="cpu").manual_seed(seed)
	else:
		generator = torch.Generator().manual_seed(seed)

	training_data = TensorDataset(torch.from_numpy(x_fit), torch.from_numpy(y_fit_array))
	training_loader = DataLoader(
		training_data, batch_size=32, shuffle=True, generator=generator
	)
	validation_features = torch.from_numpy(x_validation).to(device)
	validation_labels = torch.from_numpy(y_validation_array).to(device)
	network = TorchBreastCancerNet(x_fit.shape[1]).to(device)
	positive_count = max(float(y_fit_array.sum()), 1.0)
	negative_count = max(float(len(y_fit_array) - y_fit_array.sum()), 1.0)
	loss_function = nn.BCEWithLogitsLoss(
		pos_weight=torch.tensor([negative_count / positive_count], device=device)
	)
	optimizer = torch.optim.AdamW(network.parameters(), lr=0.001, weight_decay=0.001)
	best_state = copy.deepcopy(network.state_dict())
	best_loss = float("inf")
	best_epoch = 0
	patience = 25
	stale_epochs = 0

	for epoch in range(1, 251):
		network.train()
		for batch_features, batch_labels in training_loader:
			batch_features = batch_features.to(device)
			batch_labels = batch_labels.to(device)
			optimizer.zero_grad(set_to_none=True)
			loss = loss_function(network(batch_features), batch_labels)
			loss.backward()
			optimizer.step()

		network.eval()
		with torch.no_grad():
			validation_loss = loss_function(
				network(validation_features), validation_labels
			).item()
		if validation_loss < best_loss - 1e-5:
			best_loss = validation_loss
			best_epoch = epoch
			best_state = copy.deepcopy(network.state_dict())
			stale_epochs = 0
		else:
			stale_epochs += 1
			if stale_epochs >= patience:
				break

	network.load_state_dict(best_state)
	network.eval()
	return {
		"network": network,
		"imputer": imputer,
		"scaler": scaler,
		"feature_names": list(features.columns),
		"device": device,
		"best_epoch": best_epoch,
	}


def predict_torch_model(bundle: dict[str, object], features: pd.DataFrame) -> np.ndarray:
	ordered_features = features[bundle["feature_names"]]
	prepared = bundle["scaler"].transform(
		bundle["imputer"].transform(ordered_features)
	).astype(np.float32)
	device = bundle["device"]
	network = bundle["network"]
	network.eval()
	with torch.no_grad():
		logits = network(torch.from_numpy(prepared).to(device))
		return torch.sigmoid(logits).cpu().numpy()


def cross_validate_torch(
	features: pd.DataFrame,
	labels: pd.Series,
	cross_validator: StratifiedKFold,
	device: torch.device,
) -> dict[str, np.ndarray]:
	fold_results = []
	for fold, (training_indices, validation_indices) in enumerate(
		cross_validator.split(features, labels), start=1
	):
		training_features = features.iloc[training_indices]
		training_labels = labels.iloc[training_indices]
		validation_features = features.iloc[validation_indices]
		validation_labels = labels.iloc[validation_indices]
		bundle = fit_torch_model(
			training_features, training_labels, device, RANDOM_STATE + fold
		)
		probabilities = predict_torch_model(bundle, validation_features)
		predictions = (probabilities >= 0.5).astype(int)
		fold_results.append(
			{
				"accuracy": accuracy_score(validation_labels, predictions),
				"precision": precision_score(validation_labels, predictions),
				"recall": recall_score(validation_labels, predictions),
				"f1": f1_score(validation_labels, predictions),
				"roc_auc": roc_auc_score(validation_labels, probabilities),
			}
		)
	return {
		f"test_{metric}": np.array([fold_result[metric] for fold_result in fold_results])
		for metric in SCORING
	}


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


def load_original_dataset() -> tuple[pd.DataFrame, pd.Series]:
	if not ORIGINAL_DATA_PATH.exists():
		raise FileNotFoundError(f"Dataset not found: {ORIGINAL_DATA_PATH}")

	columns = [
		"sample_id",
		"clump_thickness",
		"cell_size_uniformity",
		"cell_shape_uniformity",
		"marginal_adhesion",
		"epithelial_cell_size",
		"bare_nuclei",
		"bland_chromatin",
		"normal_nucleoli",
		"mitoses",
		"class",
	]
	data = pd.read_csv(ORIGINAL_DATA_PATH, header=None, names=columns, na_values="?")
	labels = data["class"].map({2: 0, 4: 1})
	if labels.isna().any():
		raise ValueError("UCI class values must be 2 (benign) or 4 (malignant).")

	features = data.drop(columns=["sample_id", "class"])
	if len(data) != 699 or features.isna().sum().sum() != 16:
		raise ValueError("Unexpected shape or missing-value count in the UCI Original dataset.")
	return features, labels.astype(int)


def load_datasets() -> list[tuple[str, str, pd.DataFrame, pd.Series]]:
	wdbc_features, wdbc_labels = load_dataset()
	original_features, original_labels = load_original_dataset()
	return [
		("wdbc", "Wisconsin Diagnostic", wdbc_features, wdbc_labels),
		("original", "Wisconsin Original", original_features, original_labels),
	]


def save_evaluation_chart(
	probabilities_by_model: dict[str, np.ndarray],
	y_test: pd.Series,
	selected_predictions: np.ndarray,
	selected_name: str,
	output_path: Path,
) -> None:
	figure, (roc_axis, matrix_axis) = plt.subplots(1, 2, figsize=(12, 5))

	for name, probabilities in probabilities_by_model.items():
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
	figure.savefig(output_path, dpi=180, bbox_inches="tight")
	plt.close(figure)


def save_network_diagram(
	model: TorchBreastCancerNet,
	feature_count: int,
	dataset_name: str,
	output_path: Path,
) -> None:
	linear_layers = [layer for layer in model.modules() if isinstance(layer, nn.Linear)]
	layer_sizes = [feature_count] + [layer.out_features for layer in linear_layers]
	layer_names = ["Input features"] + [
		f"Hidden layer {index}" for index in range(1, len(layer_sizes) - 1)
	] + ["Output"]
	x_positions = np.arange(len(layer_sizes)) * 2.1
	y_positions = [np.linspace(-1, 1, size) if size > 1 else np.array([0.0]) for size in layer_sizes]

	figure, axis = plt.subplots(figsize=(14, 9))
	for layer_index, layer in enumerate(linear_layers):
		weights = layer.weight.detach().cpu().numpy().T
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

	axis.set_title(f"{dataset_name}: PyTorch network architecture", fontsize=16, pad=18)
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
	figure.savefig(output_path, dpi=180, bbox_inches="tight")
	plt.close(figure)


def save_torch_checkpoint(bundle: dict[str, object], output_path: Path) -> None:
	network = bundle["network"]
	checkpoint = {
		"state_dict": {
			name: tensor.detach().cpu() for name, tensor in network.state_dict().items()
		},
		"input_size": len(bundle["feature_names"]),
		"feature_names": bundle["feature_names"],
		"imputer_statistics": bundle["imputer"].statistics_.tolist(),
		"scaler_mean": bundle["scaler"].mean_.tolist(),
		"scaler_scale": bundle["scaler"].scale_.tolist(),
		"best_epoch": bundle["best_epoch"],
		"threshold": 0.5,
	}
	torch.save(checkpoint, output_path)


def load_torch_checkpoint(
	checkpoint_path: Path,
	device: torch.device,
) -> tuple[dict[str, object], TorchBreastCancerNet]:
	checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
	network = TorchBreastCancerNet(checkpoint["input_size"]).to(device)
	network.load_state_dict(checkpoint["state_dict"])
	network.eval()
	return checkpoint, network


def predict_checkpoint(
	checkpoint: dict[str, object],
	network: TorchBreastCancerNet,
	features: pd.DataFrame,
) -> np.ndarray:
	ordered_features = features[checkpoint["feature_names"]]
	values = ordered_features.to_numpy(dtype=np.float32)
	medians = np.asarray(checkpoint["imputer_statistics"], dtype=np.float32)
	values = np.where(np.isnan(values), medians, values)
	means = np.asarray(checkpoint["scaler_mean"], dtype=np.float32)
	scales = np.asarray(checkpoint["scaler_scale"], dtype=np.float32)
	values = ((values - means) / scales).astype(np.float32)
	device = next(network.parameters()).device
	with torch.no_grad():
		logits = network(torch.from_numpy(values).to(device))
		return torch.sigmoid(logits).cpu().numpy()


def evaluate_dataset(
	dataset_key: str,
	dataset_name: str,
	features: pd.DataFrame,
	labels: pd.Series,
	device: torch.device,
	cross_validator: StratifiedKFold,
) -> list[dict[str, object]]:
	x_train, x_test, y_train, y_test = train_test_split(
		features,
		labels,
		test_size=0.2,
		random_state=RANDOM_STATE,
		stratify=labels,
	)
	logistic_model = make_pipeline(
		SimpleImputer(strategy="median"),
		StandardScaler(),
		LogisticRegression(max_iter=3000),
	)
	logistic_scores = cross_validate(
		logistic_model, x_train, y_train, cv=cross_validator, scoring=SCORING
	)
	logistic_model.fit(x_train, y_train)

	torch_scores = cross_validate_torch(x_train, y_train, cross_validator, device)
	torch_bundle = fit_torch_model(x_train, y_train, device, RANDOM_STATE)
	probabilities_by_model = {
		"Logistic regression": logistic_model.predict_proba(x_test)[:, 1],
		"PyTorch deep network": predict_torch_model(torch_bundle, x_test),
	}
	cross_validation_scores = {
		"Logistic regression": logistic_scores,
		"PyTorch deep network": torch_scores,
	}
	cross_validation_accuracy = {}
	for name, scores in cross_validation_scores.items():
		means = {
			metric: float(np.mean(scores[f"test_{metric}"]))
			for metric in SCORING
		}
		deviations = {
			metric: float(np.std(scores[f"test_{metric}"]))
			for metric in SCORING
		}
		cross_validation_accuracy[name] = means["accuracy"]
		print(
			f"  {name}: accuracy={means['accuracy']:.3f} +/- {deviations['accuracy']:.3f}, "
			f"recall={means['recall']:.3f} +/- {deviations['recall']:.3f}, "
			f"ROC-AUC={means['roc_auc']:.3f} +/- {deviations['roc_auc']:.3f}"
		)

	selected_name = max(cross_validation_accuracy, key=cross_validation_accuracy.get)
	metrics_rows = []
	for name, probabilities in probabilities_by_model.items():
		predictions = (probabilities >= 0.5).astype(int)
		cv_scores = cross_validation_scores[name]
		metrics_rows.append(
			{
				"dataset": dataset_key,
				"model": name,
				"cv_accuracy_mean": cross_validation_accuracy[name],
				"cv_accuracy_std": np.std(cv_scores["test_accuracy"]),
				"cv_recall_mean": np.mean(cv_scores["test_recall"]),
				"cv_roc_auc_mean": np.mean(cv_scores["test_roc_auc"]),
				"test_accuracy": accuracy_score(y_test, predictions),
				"test_precision": precision_score(y_test, predictions),
				"test_recall": recall_score(y_test, predictions),
				"test_f1": f1_score(y_test, predictions),
				"test_roc_auc": roc_auc_score(y_test, probabilities),
			}
		)

	print(f"\n{dataset_name}: {len(features)} samples, {features.shape[1]} features")
	print(f"Selected by training CV: {selected_name}")
	print("Held-out test results:")
	for row in metrics_rows:
		print(
			f"  {row['model']}: accuracy={row['test_accuracy']:.3f}, "
			f"precision={row['test_precision']:.3f}, recall={row['test_recall']:.3f}, "
			f"F1={row['test_f1']:.3f}, ROC-AUC={row['test_roc_auc']:.3f}"
		)
	predictions = (probabilities_by_model[selected_name] >= 0.5).astype(int)
	print("Classification report:")
	print(classification_report(y_test, predictions, target_names=["Benign", "Malignant"]))

	suffix = "" if dataset_key == "wdbc" else f"_{dataset_key}"
	save_evaluation_chart(
		probabilities_by_model,
		y_test,
		predictions,
		selected_name,
		OUTPUT_DIR / f"model_evaluation{suffix}.png",
	)
	save_network_diagram(
		torch_bundle["network"],
		features.shape[1],
		dataset_name,
		OUTPUT_DIR / f"neural_network{suffix}.png",
	)
	joblib.dump(logistic_model, OUTPUT_DIR / f"{dataset_key}_logistic.joblib")
	save_torch_checkpoint(
		torch_bundle, OUTPUT_DIR / f"{dataset_key}_pytorch_checkpoint.pt"
	)
	return metrics_rows


def main() -> None:
	parser = argparse.ArgumentParser(description="Benchmark breast cancer classifiers.")
	parser.add_argument(
		"--device", choices=("auto", "cuda", "cpu"), default="auto"
	)
	arguments = parser.parse_args()
	device = resolve_device(arguments.device)
	torch.set_num_threads(min(torch.get_num_threads(), 4))
	OUTPUT_DIR.mkdir(exist_ok=True)
	print(f"PyTorch {torch.__version__} | device: {device}")
	if device.type == "cuda":
		print(f"GPU: {torch.cuda.get_device_name(device)}")

	cross_validator = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
	all_metrics = []
	for dataset_key, dataset_name, features, labels in load_datasets():
		all_metrics.extend(
			evaluate_dataset(
				dataset_key,
				dataset_name,
				features,
				labels,
				device,
				cross_validator,
			)
		)
	metrics_path = OUTPUT_DIR / "model_metrics.csv"
	pd.DataFrame(all_metrics).to_csv(metrics_path, index=False, float_format="%.4f")
	print(f"\nSaved {metrics_path}")


if __name__ == "__main__":
	main()
