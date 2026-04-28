import csv
import json
from pathlib import Path


def csv_to_qa_json(
    csv_path,
    json_path=None,
    course_id="cogsci_c127"
):
    """
    Convert a lecture QA CSV into a JSON list.

    Expected CSV headers:
    - Lec #
    - Lecture video name
    - Video
    - Timestamp start
    - Timestamp end
    - Question
    - Expected answer

    Video column convention:
    - 1 -> video_only
    - 0 -> speech
    """

    csv_path = Path(csv_path)

    qa_items = []

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        for i, row in enumerate(reader):
            video_value = str(row.get("Video", "")).strip()

            if video_value == "1":
                answerability_type = "video_only"
            else:
                answerability_type = "speech"

            item = {
                "course_id": course_id,
                "index": i,
                "lecture #": int(row["Lec #"]),
                "lecture_name": row["Lecture video name"].strip(),
                "Timestamp start": row["Timestamp start"].strip(),
                "Timestamp end": row["Timestamp end"].strip(),
                "question": row["Question"].strip(),
                "answer": row["Expected answer"].strip(),
                "answerability_type": answerability_type,
            }

            qa_items.append(item)

    if json_path is not None:
        json_path = Path(json_path)
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(qa_items, f, indent=2, ensure_ascii=False)

    return qa_items

if __name__ == "__main__":
    qa_data = csv_to_qa_json(
            "helpers/RAG_dataset.csv",
            "retrieval_corpus/cogsci_c127/lecture_questions.json"
        )
    print(qa_data)