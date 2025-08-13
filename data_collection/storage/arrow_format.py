import pyarrow as pa
import pyarrow.parquet as pq
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from datetime import datetime
import logging
from pathlib import Path
import json
class ArrowStorage:
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.logger = logging.getLogger(__name__)
        self._ensure_directories()
    def _ensure_directories(self):
        """Create necessary directories if they don't exist"""
        self.base_path.mkdir(parents=True, exist_ok=True)
    def _get_schema(self, data: Dict[str, Any]) -> pa.Schema:
        """Generate Arrow schema from data structure"""
        fields = []
        # Common fields
        fields.extend([
            pa.field('timestamp', pa.timestamp('ns')),
            pa.field('symbol', pa.string()),
            pa.field('open', pa.float64()),
            pa.field('high', pa.float64()),
            pa.field('low', pa.float64()),
            pa.field('close', pa.float64()),
            pa.field('volume', pa.float64())
        ])
        # Additional fields if present
        if 'vwap' in data:
            fields.append(pa.field('vwap', pa.float64()))
        if 'trades' in data:
            fields.append(pa.field('trades', pa.int64()))
        return pa.schema(fields)
    def write_ohlcv(self,
                    data: Dict[str, Any],
                    symbol: str,
                    timeframe: str,
                    compression: str = 'ZSTD',
                    compression_level: int = 3) -> bool:
        """
        Write OHLCV data to Arrow format with optimized compression
        """
        try:
            # Convert data to Arrow table
            df = pd.DataFrame(data)
            table = pa.Table.from_pandas(df, schema=self._get_schema(data))
            # Generate filename
            filepath = self.base_path / filename
            # Write with compression
            pq.write_table(
                table,
                filepath,
                compression=compression,
                compression_level=compression_level
            )
            self.logger.info(f"Successfully wrote data to {filepath}")
            return True
        except Exception as e:
            self.logger.error(f"Error writing Arrow data: {str(e)}")
            return False
    def read_ohlcv(self,
                   symbol: str,
                   timeframe: str,
                   start_date: Optional[datetime] = None,
                   end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Read OHLCV data from Arrow format with filtering
        """
        try:
            # Find matching files
            pattern = f"{symbol}_{timeframe}_*.parquet"
            files = list(self.base_path.glob(pattern))
            if not files:
                raise FileNotFoundError(f"No data found for {symbol} {timeframe}")
            # Read and concatenate data
            dfs = []
            for file in files:
                table = pq.read_table(file)
                df = table.to_pandas()
                # Apply date filters if specified
                if start_date:
                    df = df[df['timestamp'] >= start_date]
                if end_date:
                    df = df[df['timestamp'] <= end_date]
                dfs.append(df)
            return pd.concat(dfs) if dfs else pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Error reading Arrow data: {str(e)}")
            return pd.DataFrame()
    def optimize_storage(self, 
                        min_compression_ratio: float = 1.5,
                        max_file_size_mb: int = 100) -> bool:
        """
        Optimize storage by recompressing and merging files
        """
        try:
            for symbol_file in self.base_path.glob("*.parquet"):
                file_size = symbol_file.stat().st_size / (1024 * 1024)  # Size in MB
                if file_size > max_file_size_mb:
                    # Read large file
                    table = pq.read_table(symbol_file)
                    # Split into smaller chunks
                    num_chunks = int(file_size / max_file_size_mb) + 1
                    rows_per_chunk = len(table) // num_chunks
                    for i in range(num_chunks):
                        start_idx = i * rows_per_chunk
                        end_idx = start_idx + rows_per_chunk
                        chunk = table.slice(start_idx, end_idx)
                        # Write optimized chunk
                        new_filename = f"{symbol_file.stem}_part{i}{symbol_file.suffix}"
                        pq.write_table(
                            chunk,
                            self.base_path / new_filename,
                            compression='ZSTD',
                            compression_level=5
                        )
                    # Remove original large file
                    symbol_file.unlink()
            return True
        except Exception as e:
            self.logger.error(f"Error optimizing storage: {str(e)}")
            return False
