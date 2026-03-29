# -*- coding: utf-8 -*-
"""
Hyperspectral Data Preprocessing and Feature Selection Pipeline
This script performs Multiplicative Scatter Correction (MSC) and
Competitive Adaptive Reweighted Sampling (CARS) for hyperspectral data analysis.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
import warnings

# Suppress warnings for cleaner console output
warnings.filterwarnings('ignore')

# ==========================================
# Configuration (配置项：统一管理路径)
# ==========================================
# Use relative paths for GitHub repository
INPUT_FILE = './data/dataLAI4.xlsx'
OUTPUT_DIR = './output/'


def msc(sdata):
    """
    Perform Multiplicative Scatter Correction (MSC) on hyperspectral data.
    """
    n = sdata.shape[0]  # Number of samples
    k = np.zeros(sdata.shape[0])
    b = np.zeros(sdata.shape[0])

    M = np.mean(sdata, axis=0)

    for i in range(n):
        y = sdata[i, :]
        y = y.reshape(-1, 1)
        M_reshaped = M.reshape(-1, 1)
        model = LinearRegression()
        model.fit(M_reshaped, y)
        k[i] = model.coef_[0, 0]
        b[i] = model.intercept_[0]

    spec_msc = np.zeros_like(sdata)
    for i in range(n):
        bb = np.repeat(b[i], sdata.shape[1])
        kk = np.repeat(k[i], sdata.shape[1])
        temp = (sdata[i, :] - bb) / kk
        spec_msc[i, :] = temp
    return spec_msc


def CARS(X, y, wavelengths, iteration=20, n_comps=100, cv=5):
    """
    Competitive Adaptive Reweighted Sampling (CARS) for feature selection.
    """
    N, D = X.shape
    prob = 0.8
    a = np.power((D / 2), (1 / (iteration - 1)))
    k = (np.log(D / 2)) / (iteration - 1)
    r = [round(a * np.exp(-(k * i)) * D) for i in range(1, iteration + 1)]

    weights = np.ones(D) / D
    RMSECV = []
    idWs = []

    for i in range(iteration):
        idCal = np.random.choice(np.arange(N), size=int(prob * N), replace=False)
        idW = np.random.choice(np.arange(D), size=r[i], p=weights / weights.sum(), replace=False)
        idWs.append(idW)

        X_cal = X[idCal[:, np.newaxis], idW]
        Y_cal = y[idCal]
        comp = min(n_comps, len(idW))
        pls = PLSRegression(n_components=comp)
        pls.fit(X_cal, Y_cal)

        absolute = np.abs(pls.coef_).reshape(-1)
        # Added 1e-8 to prevent division by zero warning
        weights[idW] = absolute / (sum(absolute) + 1e-8)

        MSE = -cross_val_score(pls, X_cal, Y_cal, cv=cv, scoring="neg_mean_squared_error")
        RMSE = np.mean(np.sqrt(MSE))
        RMSECV.append(RMSE)

    best_index = np.argmin(RMSECV)
    W_best_indices = idWs[best_index]
    W_best_columns = wavelengths[W_best_indices]
    return W_best_columns


def main():
    # 1. Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Output directory ready.")

    # 2. Load and clean data
    print(f"Loading data from {INPUT_FILE}...")
    data = pd.read_excel(INPUT_FILE)

    # Safely fill missing values in the second column (LAI/Yield label)
    label_col = data.columns[1]
    data[label_col] = data[label_col].fillna(data[label_col].mean())
    data_cleaned = data.dropna().reset_index(drop=True)

    # 3. Apply MSC
    print("Applying Multiplicative Scatter Correction (MSC)...")
    spectra = data_cleaned.iloc[:, 2:].values
    wavelengths = data_cleaned.columns[2:]
    msc_spectra = msc(spectra)

    # 4. Plot and save spectra comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].plot(wavelengths, spectra.T)
    axes[0].set_title('Original Spectra', fontsize='x-large', fontname='Arial', fontweight='bold')
    axes[0].set_xlabel("Wavenumber(nm)", fontname='Arial', fontweight='bold')
    axes[0].set_ylabel("Reflectance-Org", fontname='Arial', fontweight='bold')

    axes[1].plot(wavelengths, msc_spectra.T)
    axes[1].set_title('MSC Spectra', fontname='Arial', fontweight='bold', fontsize='x-large')
    axes[1].set_xlabel("Wavenumber(nm)", fontname='Arial', fontweight='bold')
    axes[1].set_ylabel("Reflectance-MSC", fontname='Arial', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'spectra_comparisonLAI.jpg'))
    plt.savefig(os.path.join(OUTPUT_DIR, 'spectra_comparison.svg'), dpi=300)
    print("Spectra comparison plots saved.")

    # 5. Prepare MSC data for output
    metadata = data_cleaned.iloc[:, :2]
    firstline = pd.DataFrame(data_cleaned.iloc[:, 2:])
    msc_data = pd.DataFrame(msc_spectra, columns=firstline.columns)
    msc_results = pd.concat([metadata, msc_data], axis=1)
    msc_results.to_csv(os.path.join(OUTPUT_DIR, "msc_resultsLAI.csv"), index=False)

    # 6. Apply CARS feature selection
    print("Running CARS feature selection (this may take a moment)...")
    X_org = data_cleaned.iloc[:, 2:].values
    X_msc = msc_results.iloc[:, 2:].values
    y = metadata.iloc[:, 1].values

    Org_selected_features = CARS(X_org, y, wavelengths)
    MSC_selected_features = CARS(X_msc, y, wavelengths)

    # 7. Save CARS selected features
    Org_X_selected = data_cleaned.iloc[:, 2:][list(Org_selected_features)].reset_index(drop=True)
    MSC_X_selected = data_cleaned.iloc[:, 2:][list(MSC_selected_features)].reset_index(drop=True)

    org_select = pd.concat([metadata, Org_X_selected], axis=1)
    org_select.to_csv(os.path.join(OUTPUT_DIR, "org_selectLAI.csv"), index=False)

    MSC_select = pd.concat([metadata, MSC_X_selected], axis=1)
    MSC_select.to_csv(os.path.join(OUTPUT_DIR, "MSC_selectLAI.csv"), index=False)

    print("Pipeline completed successfully! All results are saved in the output folder.")


if __name__ == "__main__":
    main()