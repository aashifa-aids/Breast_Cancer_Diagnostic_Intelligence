from src.pipeline import run_pipeline

if __name__ == "__main__":
    results = run_pipeline()
    print(results.to_string(index=False))
