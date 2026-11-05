def clean_dataset(frame):
    cleaned = frame.copy()
    report = {
        "rows_before": int(len(cleaned)),
        "columns": int(cleaned.shape[1]),
        "missing_cells_before": int(cleaned.isna().sum().sum()),
        "duplicate_rows_before": int(cleaned.duplicated().sum())
    }
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    report.update({
        "rows_after": int(len(cleaned)),
        "missing_cells_after": int(cleaned.isna().sum().sum()),
        "duplicate_rows_after": int(cleaned.duplicated().sum())
    })
    return cleaned, report
