{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyMVmxxF5jznMch6HuuGiEa5",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Ashokyadhav/Econocausal/blob/Saiesh/validate_dataset.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import argparse\n",
        "import pandas as pd\n",
        "\n",
        "def main():\n",
        "    parser = argparse.ArgumentParser()\n",
        "    parser.add_argument(\"--csv_path\", help=\"Path to the dataset CSV\", default=\"dataset.csv\")\n",
        "    parser.add_argument(\"--chunksize\", type=int, default=250_000)\n",
        "\n",
        "    # Colab kernel conflict raakunda undadaniki idi use chestunnam\n",
        "    args, unknown = parser.parse_known_args()\n",
        "\n",
        "    total_rows = 0\n",
        "    missing = {}\n",
        "    dtypes = {}\n",
        "    minima = {}\n",
        "    maxima = {}\n",
        "    chunk_local_duplicates = 0\n",
        "    columns = None\n",
        "\n",
        "    try:\n",
        "        for chunk in pd.read_csv(args.csv_path, chunksize=args.chunksize):\n",
        "            total_rows += len(chunk)\n",
        "            if columns is None:\n",
        "                columns = list(chunk.columns)\n",
        "            if list(chunk.columns) != columns:\n",
        "                raise ValueError(\"Column names/order changed between chunks\")\n",
        "\n",
        "            for col in chunk.columns:\n",
        "                missing[col] = missing.get(col, 0) + int(chunk[col].isna().sum())\n",
        "                dtypes.setdefault(col, str(chunk[col].dtype))\n",
        "                if pd.api.types.is_numeric_dtype(chunk[col]):\n",
        "                    lo, hi = chunk[col].min(), chunk[col].max()\n",
        "                    minima[col] = min(minima.get(col, lo), lo)\n",
        "                    maxima[col] = max(maxima.get(col, hi), hi)\n",
        "\n",
        "            chunk_local_duplicates += int(chunk.duplicated().sum())\n",
        "\n",
        "        print(f\"Rows: {total_rows:,}\")\n",
        "        print(f\"Columns ({len(columns or [])}): {columns}\")\n",
        "        print(\"\\nData types:\")\n",
        "        for col, dtype in dtypes.items():\n",
        "            print(f\"  {col}: {dtype}\")\n",
        "        print(\"\\nMissing values:\")\n",
        "        for col, count in missing.items():\n",
        "            print(f\"  {col}: {count:,}\")\n",
        "        print(f\"\\nExact duplicate rows within chunks: {chunk_local_duplicates:,}\")\n",
        "        print(\"Note: duplicates spanning different chunks are not detected.\")\n",
        "        print(\"\\nNumeric ranges:\")\n",
        "        for col in minima:\n",
        "            print(f\"  {col}: min={minima[col]}, max={maxima[col]}\")\n",
        "\n",
        "    except FileNotFoundError:\n",
        "        print(f\"Error: The file '{args.csv_path}' was not found. Please verify the path.\")\n",
        "\n",
        "if __name__ == \"__main__\":\n",
        "    main()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "f5xmvNOujlyb",
        "outputId": "5a2c060d-8bf4-4a25-c9da-6b501eb18eee"
      },
      "execution_count": 8,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Error: The file 'dataset.csv' was not found. Please verify the path.\n"
          ]
        }
      ]
    }
  ]
}