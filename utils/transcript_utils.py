from pathlib import Path
import re

TIMESTAMP_PATTERN = re.compile(r"^\d{2}:\d{2}:\d{2}\.\d{3}\s+-->\s+\d{2}:\d{2}:\d{2}\.\d{3}$")


def clean_transcript_text(transcript: str) -> str:
    """Remove timestamp lines from a transcript and return only the spoken text."""
    lines = transcript.splitlines()
    cleaned_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if TIMESTAMP_PATTERN.match(stripped):
            continue
        cleaned_lines.append(stripped)

    return "\n".join(cleaned_lines)


def load_and_clean_transcripts(original_dir: Path | str,
                               new_dir: Path | str):
    ORIGINAL_PATH = Path(original_dir)
    NEW_PATH = Path(new_dir)
    
    # create the new directory
    NEW_PATH.mkdir(parents=True, exist_ok=True)
    
    transcripts = ORIGINAL_PATH.glob(pattern="*txt")
    for transcript_path in transcripts:
        # read the text file
        file_content = transcript_path.read_text(encoding="utf-8")
        # clean the text
        cleaned_content = clean_transcript_text(file_content)
        # filename
        filename = transcript_path.name
        # create the output path
        output_path = NEW_PATH / filename
        # write the file content
        output_path.write_text(data=cleaned_content,
                               encoding="utf-8")


if __name__ == "__main__":
    REPO_ROOT = Path(__file__).resolve().parent.parent

    load_and_clean_transcripts(REPO_ROOT / "data" / "raw",
                               REPO_ROOT / "data" / "processed")