# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "coverage",
#     "invoke",
#     "joblib",
#     "pandarallel",
#     "pandas",
#     "pytest",
#     "typer",
# ]
# ///
import os
from functools import partial


def check_coverage_percentage(source_code, test_case):
    import tempfile

    from invoke import run

    # with temprorary directory
    if not test_case:
        return "0%"
    outfile_text = f"""
{source_code}

{test_case}
    """
    k = 0
    with tempfile.TemporaryDirectory() as temp_dir:
        file_name = f"mytest_{k}.py"
        outfile_name = f"{temp_dir}/{file_name}"
        with open(outfile_name, "w") as f:
            f.write(outfile_text)
        print(f"cd {temp_dir} && coverage run -m pytest {file_name} && cd -")
        run(f"cd {temp_dir} && coverage run -m pytest {file_name}")
        coverage_output = run(f"cd {temp_dir} && coverage report ")
        percentage = coverage_output.stdout.splitlines()[-1].split()[-1]
        return percentage


def annotate_best_test_from_coverage(row):
    try:
        code_text = row["code"]
        t = row["test_cases"]
        pcs = [(k, check_coverage_percentage(code_text, v)) for k, v in t.items()]
        sorted_list = sorted(pcs, key=lambda x: int(x[1].replace("%", "")))
        found_best = None
        found_best_index = -1
        test_sorted_list = sorted_list.copy()
        while not found_best:
            test_entry = test_sorted_list.pop()
            test_index = test_entry[0]
            if not (
                row["components"]["test_cases_components"][test_index][
                    "predicted_output"
                ]
                == "unknown"
                or row["components"]["test_cases_components"][test_index][
                    "predicted_input"
                ]
                == "unknown"
            ):
                found_best = True
                found_best_index = test_index
        row["best_test_case_coverage"] = found_best_index
        row["coverages"] = str(sorted_list)
    except Exception as e:
        row["best_test_case_coverage"] = -1
        row["coverages"] = f"Failed: {e}"
    return row


def process_file(input_file, output_file):
    import pandas as pd

    df = pd.read_json(input_file, lines=True)
    df_out = df.parallel_apply(annotate_best_test_from_coverage, axis=1)
    df_out.to_json(output_file, lines=True, orient="records")


def process_folder(input_folder, output_folder, dry_run=False):
    import glob

    files = glob.glob(f"{input_folder}/*.jsonl")
    from joblib import Parallel, delayed

    dff = pd.DataFrame({"files": files})

    def parallel_process_files(entry, outfolder):
        file_name = entry
        file_base_name = os.path.basename(file_name)
        output_file_name = os.path.join(
            outfolder, file_base_name.replace(".jsonl", "_annotated.jsonl")
        )
        print("processing", file_base_name, output_file_name)
        if not dry_run:
            process_file(file_name, output_file_name)

    parallel_process_files_ = partial(parallel_process_files, outfolder=output_folder)
    results_threads = Parallel(n_jobs=min(len(files), 6), backend="threading")(
        delayed(parallel_process_files_)(f) for f in files
    )


def main(input_folder: str, output_folder: str, dry_run: bool = False):
    from pandarallel import pandarallel

    pandarallel.initialize(progress_bar=True, nb_workers=5)
    process_folder(input_folder, output_folder, dry_run)


if _name__ == "__main__":
    import typer

    typer.run(main)
