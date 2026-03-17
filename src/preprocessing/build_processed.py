from src.models.collaborative import clear_collaborative_cache
from src.models.content_based import clear_content_cache
from src.preprocessing.artifacts import build_processed_artifacts


def main():
    summary = build_processed_artifacts()
    clear_content_cache()
    clear_collaborative_cache()

    print("Processed artifacts created successfully.")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
