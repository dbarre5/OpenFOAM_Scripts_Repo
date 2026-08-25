import subprocess
import optuna
import os

CSV_HEADER = (
    "featureAngle,nGrow,nRelaxIter,"
    "nSmoothSurfaceNormals,nSmoothNormals,nSmoothThickness,coverage\n"
)

try:
    open("results/runs.csv", "x").write(CSV_HEADER)
except FileExistsError:
    pass


def objective(trial):
    params = {
        "featureAngle": trial.suggest_int("featureAngle", 90, 150),
        "nGrow": trial.suggest_int("nGrow", 0, 3),
        "nRelaxIter": trial.suggest_int("nRelaxIter", 1, 8),
        "nSmoothSurfaceNormals": trial.suggest_int("nSmoothSurfaceNormals", 0, 3),
        "nSmoothNormals": trial.suggest_int("nSmoothNormals", 0, 5),
        "nSmoothThickness": trial.suggest_int("nSmoothThickness", 1, 20),
    }

    env = os.environ.copy()
    env.update({k: str(v) for k, v in params.items()})

    try:
        result = subprocess.check_output(
            ["bash", "run_snappy.sh"],
            env=env,
            timeout=3600,
            text=True,
        )
        return float(result.strip())

    except Exception as e:
        print("❌ snappy failed:", e)
        return 0.0


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

print("Best coverage:", study.best_value)
print("Best params:", study.best_params)

