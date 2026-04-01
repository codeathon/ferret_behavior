import pandas as pd
from pathlib import Path

from python_code.utilities.logging_config import get_logger

logger = get_logger(__name__)

if __name__=='__main__':
    folder_path = Path("/home/scholl-lab/recordings/session_2025-04-28/ferret_9C04_NoImplant_P35_E3/skellyclicker_data")
    logger.info("Checking for negative values in %s", folder_path)
    csv_files = sorted(list(folder_path.iterdir()))

    for csv in csv_files:
        df = pd.read_csv(str(csv))
        df_numeric = df.apply(pd.to_numeric, errors='coerce')
        negative_rows = df_numeric[df_numeric.lt(0).any(axis=1)]
        logger.info("Negative rows in %s:\n%s", csv, negative_rows)