# AI Fundraising Preparation

Synthetic donor dataset and a small KPI report to support fundraising analysis.

## Structure

```
ai-fundraising/
├── data/
│   └── donors.csv              # 100 synthetic donors
├── notebooks/
│   └── donor_analysis.ipynb    # exploratory analysis
├── src/
│   ├── generate_donors.py      # builds data/donors.csv
│   └── donor_kpis.py           # prints donor KPIs
└── requirements.txt
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run all commands from the project root (scripts use the relative path `data/donors.csv`).

Generate the dataset:

```bash
python src/generate_donors.py
```

Print the KPI report:

```bash
python src/donor_kpis.py                    # uses data/donors.csv
python src/donor_kpis.py path/to/other.csv  # any donors file
```

## Data

`data/donors.csv` — one row per donor:

| Column | Meaning |
|---|---|
| `donor_id` | Donor ID, `D001`–`D100` |
| `total_donation` | Total amount given ($) |
| `donation_count` | Number of gifts |
| `days_since_last_donation` | Days since most recent gift |

Data is random (seed 42), so each run produces the same file.

`donor_kpis.py` also reads the display format
(`Donor, Total Donation, Donations, Last Donation` with values like `$1,000` and `10 days ago`).

## KPIs

| KPI | Definition |
|---|---|
| Total Donors | Number of donors |
| Total Donations | Sum of `total_donation` (plus total gift count) |
| Average Donation | Total amount ÷ donors, and total amount ÷ gifts |
| Highest Donor | Donor with the largest `total_donation` |
| Inactive Donors | No gift in more than 365 days (`INACTIVE_DAYS` in `donor_kpis.py`) |
