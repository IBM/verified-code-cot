# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

# Standard
import json
import pickle


class Tools:
    @staticmethod
    def load_jsonl_v1(file_path):
        json_objects = []
        with open(file_path, "r", encoding="utf8") as f:
            for line in f:
                json_objects.append(json.loads(line.strip()))
        return json_objects

    # Standard
    import json

    def load_jsonl(file_path, max_entries=10):
        json_objects = []
        with open(file_path, "r", encoding="utf8") as f:
            for i, line in enumerate(f):
                if max_entries is not None and i >= max_entries:
                    break
                json_objects.append(json.loads(line.strip()))
        return json_objects

    @staticmethod
    def load_tasks(tasks):
        # Determine if the file is in JSON or JSONL format

        tasks = Tools.load_json(tasks)

        for task in tasks:
            if "id" in task:
                task["task_id"] = task.pop("id")

        return tasks

    @staticmethod
    def load_json(data, max_entries=10):
        if isinstance(data, dict):
            data = [data]

        # Clean the 'responses' field if it exists
        for entry in data:
            if "responses" in entry:
                entry["responses"] = [
                    Tools.clean_code(response) for response in entry["responses"]
                ]

        return data

    @staticmethod
    def clean_code(code_text):
        # Split by 'python' and keep only unique parts
        unique_code = set(part.strip() for part in code_text.split("python"))
        # Join unique parts back into a single string
        return "\n".join(unique_code)

    @staticmethod
    def dump_pickle(path, content):
        with open(path, "wb") as f:
            pickle.dump(content, f)

    @staticmethod
    def load_pickle(path):
        with open(path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def write_file(path, content):
        with open(path, "w", encoding="utf8") as f:
            f.write(content)
