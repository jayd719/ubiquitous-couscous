#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
import sys
from io import StringIO
from typing import Optional
import argparse
from staticFiles import SCRIPTS


def setup_environment(output_dir: str = "./output/") -> str:
    """Create output directory if it doesn't exist"""
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def load_data(file_path: str) -> pd.DataFrame:
    """Load dataset from specified file path"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at: {file_path}")
    return pd.read_csv(file_path)


def save_as_html(data: pd.DataFrame, filename: str, output_dir: str) -> None:
    """Save DataFrame as HTML file with sanitized filename"""
    sanitized_name = "".join(c for c in filename if c.isalnum() or c in ("-", "_"))
    file_path = os.path.join(output_dir, f"{sanitized_name}.html")
    data.to_html(file_path)
    print(f"Saved: {file_path}")


def capture_dataframe_info(df: pd.DataFrame) -> str:
    """Capture output of dataframe.info() as string"""
    buffer = StringIO()
    df.info(buf=buffer)
    return buffer.getvalue()


def generate_report_content(output_dir: str) -> str:
    """Generate HTML report from all saved analysis files"""
    html_content = []

    # Process files in numerical order
    for file_name in sorted(f for f in os.listdir(output_dir) if f.endswith(".html")):
        file_path = os.path.join(output_dir, file_name)
        section_name = os.path.splitext(file_name)[0].split("_", 1)[-1].title()

        with open(file_path, "r", encoding="utf-8") as f:
            html_content.append(f"<h2>{section_name}</h2>")
            html_content.append(f.read())

    return "\n".join(html_content)


def create_full_report(output_dir: str, report_file: str = "REPORT.html") -> None:
    """Compile final report from all analysis components"""
    report_path = os.path.join(output_dir, report_file)
    html_content = generate_report_content(output_dir)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"<html><head><title>Data Analysis Report</title></head><body><main>")
        f.write(html_content)
        f.write(f"</main>{SCRIPTS}</body></html>")
    print(f"Report generated: {report_path}")


def perform_analysis(df: pd.DataFrame, output_dir: str) -> None:
    """Main analysis pipeline"""
    # Basic description
    save_as_html(df.describe(), "01_description", output_dir)

    # Data structure info
    info_html = pd.DataFrame(
        {
            "dtype": df.dtypes,
            "non_null": df.count(),
            "null": df.isnull().sum(),
            "unique": df.nunique(),
        }
    )
    save_as_html(info_html, "02_data_structure", output_dir)

    # Sample data
    save_as_html(df.head(10), "03_data_sample", output_dir)

    # Correlation matrix
    if df.select_dtypes(include=np.number).shape[1] > 1:
        save_as_html(df.corr(), "04_correlation_matrix", output_dir)


def main(default="Assignment3/archive/test.csv"):
    parser = argparse.ArgumentParser(description="Data Analysis Report Generator")
    parser.add_argument("-i", "--input", default=default, help="Input CSV file path")
    parser.add_argument("-o", "--output", default="./output/", help="Output directory")
    args = parser.parse_args()

    output_dir = setup_environment(args.output)

    try:
        df = load_data(args.input)
        perform_analysis(df, output_dir)
        create_full_report(output_dir)
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
