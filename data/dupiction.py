import pandas as pd
import os
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def deduplicate_csv(
    input_path,
    output_path=None,
    subset=None,
    keep='first',
    create_backup=True
):
    """
    Remove duplicate rows from a CSV file.

    Args:
        input_path (str): Path to the input CSV file
        output_path (str, optional): Path to save the deduplicated CSV file. If None, will modify the original filename.
        subset (list, optional): List of column names to consider for identifying duplicates. If None, uses all columns.
        keep (str): Which duplicates to keep: 'first', 'last', or False (drop all duplicates)
        create_backup (bool): Whether to create a backup of the original file
    """
    try:
        # Create output path if not provided
        if output_path is None:
            base, ext = os.path.splitext(input_path)
            output_path = f"{base}_deduplicated{ext}"

        # Create backup if requested
        if create_backup:
            backup_path = f"{os.path.splitext(input_path)[0]}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            logger.info(f"Creating backup at {backup_path}")
            os.system(f"cp '{input_path}' '{backup_path}'")

        # Load the CSV file
        logger.info(f"Loading data from {input_path}")
        df = pd.read_csv(input_path)
        original_rows = len(df)
        logger.info(f"Original dataset: {original_rows} rows, {len(df.columns)} columns")

        # Check for and remove duplicates
        logger.info(f"Removing duplicates (keep={keep})" + (f" based on columns: {subset}" if subset else ""))
        df_unique = df.drop_duplicates(subset=subset, keep=keep)
        new_rows = len(df_unique)

        # Calculate statistics
        duplicates_removed = original_rows - new_rows
        duplicate_percentage = (duplicates_removed / original_rows) * 100 if original_rows > 0 else 0

        logger.info(f"Found {duplicates_removed} duplicate rows ({duplicate_percentage:.2f}%)")
        logger.info(f"Deduplicated dataset: {new_rows} rows")

        # Save the deduplicated data
        logger.info(f"Saving deduplicated data to {output_path}")
        df_unique.to_csv(output_path, index=False)

        return {
            'original_rows': original_rows,
            'new_rows': new_rows,
            'duplicates_removed': duplicates_removed,
            'duplicate_percentage': duplicate_percentage,
            'output_path': output_path
        }

    except Exception as e:
        logger.error(f"Error during deduplication: {str(e)}")
        raise

# Example usage
if __name__ == "__main__":
    # Use the local path since we're already in the data directory
    results = deduplicate_csv('courses_large.csv')
    print(f"Successfully removed {results['duplicates_removed']} duplicates!")
