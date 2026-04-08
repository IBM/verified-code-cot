import os

import pandas as pd
from filter_qa_based_on_score import (
    filter_conversation,
    filter_conversation_with_coverage,
)


def select_best(row):
    best_case_scored = False
    try:
        best_test_case_index_old = row["best_test_case_index"]
        best_case_scored = True
    except KeyError:
        pass
    row["best_test_case_index"] = int(row["best_test_case_coverage"])
    row2 = filter_conversation_with_coverage(row)
    if best_case_scored:
        row2["best_test_case_index"] = best_test_case_index_old
        # NOTE: in the output data, ignore the 'best_test_case_index' column
    return row2


def select_best_v2(row):
    best_case_scored = False
    try:
        best_test_case_index_old = row["best_test_case_index"]
        best_case_scored = True
    except KeyError:
        pass
    try:
        row2 = filter_conversation(row)
    except:
        row2 = row
    return row2


def process_file(input_file, output_file):
    df = pd.read_json(input_file, lines=True)
    df_out = df.parallel_apply(select_best, axis=1)
    df_out.to_json(output_file, lines=True, orient="records")


from functools import partial


def process_folder(input_folder, output_folder, dry_run=False):
    import glob

    files = glob.glob(f"{input_folder}/*.jsonl")
    from joblib import Parallel, delayed

    # print (files)
    dff = pd.DataFrame({"files": files})

    def parallel_process_files(entry, outfolder):
        file_name = entry
        file_base_name = os.path.basename(file_name)
        output_file_name = os.path.join(
            outfolder, file_base_name.replace(".jsonl", "_filtered.jsonl")
        )
        print("processing", file_base_name, output_file_name)
        if not dry_run:
            process_file(file_name, output_file_name)

    parallel_process_files_ = partial(parallel_process_files, outfolder=output_folder)
    results_threads = Parallel(n_jobs=1, backend="threading")(
        delayed(parallel_process_files_)(f) for f in files
    )


def main(input_folder: str, output_folder: str, dry_run: bool = False):
    from pandarallel import pandarallel

    pandarallel.initialize(progress_bar=True, nb_workers=5)
    process_folder(input_folder, output_folder, dry_run)


if __name__ == "__main__":
    import typer

    typer.run(main)
