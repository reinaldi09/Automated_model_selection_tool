import pandas as pd
import numpy as np

# Generate synthetic dataset for demonstration
np.random.seed(0)
X = np.random.rand(100, 1) * 10  # Feature values between 0 and 10
y = 2 * X.squeeze() + np.random.randn(100)  # Target values with noise

# Create a DataFrame
df = pd.DataFrame({'Feature': X.squeeze(), 'Target': y})

# Save DataFrame to CSV file
df.to_csv('dataset.csv', index=False)