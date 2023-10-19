import yaml
import pathlib
from datetime import datetime
from giskard import Dataset, Model, scan
from giskard.models.base import BaseModel
from giskard.ml_worker.utils.file_utils import get_file_name

from pathlib import Path
from giskard.core.core import DatasetMeta


def load_giskard_model_dataset(model_path="model", dataset_path="dataset") -> (BaseModel, Dataset):
    with open(Path(dataset_path) / "giskard-dataset-meta.yaml") as f:
        saved_meta = yaml.load(f, Loader=yaml.Loader)
        meta = DatasetMeta(
            name=saved_meta["name"],
            target=saved_meta["target"],
            column_types=saved_meta["column_types"],
            column_dtypes=saved_meta["column_dtypes"],
            number_of_rows=saved_meta["number_of_rows"],
            category_features=saved_meta["category_features"],
        )

    df = Dataset.load(Path(dataset_path) / get_file_name("data", "csv.zst", False))
    df = Dataset.cast_column_to_dtypes(df, meta.column_dtypes)

    return Model.load(model_path), Dataset(
        df=df,
        name=meta.name,
        target=meta.target,
        column_types=meta.column_types,
    )


if __name__ == '__main__':
    model, data = load_giskard_model_dataset()
    report = scan(model, data, only=["performance_bias"])
    pathlib.Path("report").mkdir(parents=True, exist_ok=True)
    date = str(datetime.now().strftime("%Y.%m.%d-%H.%M.%S"))
    report.to_html("report/"+date+".html")
    report.to_html("report/"+date+".md")
    print(date)
