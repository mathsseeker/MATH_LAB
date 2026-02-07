# Data Files

This directory contains sample datasets for use in laboratory exercises and assignments.

## Available Datasets

### production_data.csv
Manufacturing production data with quality metrics.

**Columns:**
- `ProductID`: Unique product identifier
- `DefectRate`: Defect rate (0-1 scale)
- `ProductionTime`: Time to produce (minutes)
- `MaterialCost`: Cost of materials ($)
- `LaborHours`: Labor hours required
- `QualityScore`: Quality score (0-100)

**Use cases:**
- Statistical analysis
- Quality control studies
- Correlation analysis
- Regression modeling
- Production optimization

### demand_data.csv
Daily demand and inventory data for products.

**Columns:**
- `Date`: Date of observation
- `Product`: Product identifier (A, B, C)
- `DemandQuantity`: Daily demand quantity
- `OrdersReceived`: Quantity received from suppliers
- `StockLevel`: End-of-day inventory level

**Use cases:**
- Inventory analysis
- Demand forecasting
- EOQ calculations
- Safety stock analysis
- Order pattern analysis

### machine_data.csv
Machine performance and reliability data.

**Columns:**
- `MachineID`: Machine identifier
- `OperatingHours`: Hours in operation
- `DowntimeHours`: Hours of downtime
- `MaintenanceCount`: Number of maintenance events
- `FailureCount`: Number of failures
- `Efficiency`: Operating efficiency (0-1 scale)

**Use cases:**
- Reliability analysis
- Maintenance scheduling
- Availability calculations
- Efficiency studies
- Failure prediction

## Loading Data

### Using Pandas
```python
import pandas as pd

# Load data
df = pd.read_csv('data/production_data.csv')

# View first few rows
print(df.head())

# Get summary statistics
print(df.describe())

# Filter data
high_quality = df[df['QualityScore'] > 95]
```

### Using NumPy
```python
import numpy as np

# Load numerical data (skip header)
data = np.genfromtxt('data/production_data.csv', 
                      delimiter=',', 
                      skip_header=1,
                      usecols=(1, 2, 3, 4, 5))  # Numerical columns only
```

## Data Analysis Tips

1. **Explore first**: Always check the data structure and summary statistics
2. **Check for missing values**: Look for NaN or empty values
3. **Validate ranges**: Ensure values are within expected ranges
4. **Visualize**: Create plots to understand distributions
5. **Document**: Note any assumptions or data cleaning steps

## Example Analysis

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load production data
df = pd.read_csv('data/production_data.csv')

# Analyze relationship between defect rate and quality score
plt.figure(figsize=(10, 6))
plt.scatter(df['DefectRate'], df['QualityScore'])
plt.xlabel('Defect Rate')
plt.ylabel('Quality Score')
plt.title('Quality Score vs. Defect Rate')
plt.grid(True, alpha=0.3)
plt.show()

# Calculate correlation
correlation = df[['DefectRate', 'QualityScore']].corr()
print(correlation)
```

## Creating Your Own Data

Students are encouraged to:
- Generate synthetic data for specific scenarios
- Collect real data from case studies
- Modify existing datasets for different analyses
- Share interesting datasets with the class

## Data Ethics

When working with data:
- Use only provided or publicly available data
- Respect data privacy and confidentiality
- Cite data sources appropriately
- Don't share proprietary or sensitive data

## Additional Resources

- Pandas documentation: https://pandas.pydata.org/docs/
- Data analysis tutorials: Various online resources
- Statistical analysis guides: Course materials
