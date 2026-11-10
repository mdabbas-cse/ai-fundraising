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
│   ├── donor_kpis.py           # prints donor KPIs
│   └── segment_donors.py       # assigns a segment to each donor
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

Segment donors (prints `D005 → High Value Donor` etc. and writes `data/donor_segments.csv`):

```bash
python src/segment_donors.py
```

## Data

`data/donors.csv` — one row per donor:

| Column | Meaning |
|---|---|
| `donor_id` | Donor ID, `D001`–`D100` |
| `total_donation` | Total amount given ($) |
| `donation_count` | Number of gifts |
| `days_since_last_donation` | Days since most recent gift |

Data is random (seed 42), so each run produces the same file. Rows `D001`–`D006` are fixed
reference donors; for the rest, total giving grows with the number of gifts and frequent
donors tend to have given more recently.

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

## Segments

Rules are checked top to bottom; the first match wins.

| Segment | Rule | Example |
|---|---|---|
| Inactive Donor | No gift in more than 365 days | D004 ($20, 400 days ago) |
| High Value Donor | Total giving ≥ $1,000 (`HIGH_VALUE_MIN`) | D005 ($2,000, 20 gifts) |
| New Donor | Only 1 gift so far | D002 ($50, 1 gift) |
| Regular Donor | Everyone else | D003 ($500, 5 gifts) |
