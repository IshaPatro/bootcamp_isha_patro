


## Data Storage

This stage demonstrates efficient data storage patterns using both CSV and Parquet formats with environment-driven configuration.

**Folder Structure:**
- `data/raw/` - CSV files with timestamped names
- `data/processed/` - Parquet files for optimized storage
- Environment variables control storage paths via `.env`

**Validation:** Files are reloaded and checked for shape consistency and proper data types. Parquet offers better compression and preserves data types automatically.

## Project Structure

```
├── src/stage05_data-storage_notebook.ipynb  # Storage implementation
├── data/
│   ├── raw/                    # CSV storage
│   └── processed/              # Parquet storage
└── README.md                   # Documentation
```
