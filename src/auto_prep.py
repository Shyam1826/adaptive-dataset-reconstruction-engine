"""
AutoDataPrepPipeline Module
Automated profiling, cleaning, imputation, encoding, scaling, and correlation pruning
of reconstructed datasets to prepare them for machine learning tasks.
Supports standard scikit-learn pipelines and a dual-output lifecycle.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

# Configure logger
logger = logging.getLogger("AutoDataPrepPipeline")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s - %(name)s - %(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class AutoDataPrepPipeline:
    """Production-ready automated data preprocessing pipeline.
    
    Profiles, cleans, imputes, encodes, scales, and prunes Pandas DataFrames
    to produce clean, model-ready outputs. Adheres to standard scikit-learn interfaces
    and supports a dual-output architecture yielding both cleaned and model-ready sets.
    """

    def __init__(
        self,
        target_column: Optional[str] = None,
        isolate_target: bool = True,
        skewness_threshold: float = 0.5,
        correlation_threshold: float = 0.85,
        cardinality_threshold: int = 10,
        noise_threshold: float = 0.80,
    ):
        """Initializes the preprocessing pipeline.
        
        Args:
            target_column: The name of the target column in the dataset.
            isolate_target: If True, prevents target column modifications.
            skewness_threshold: The threshold to distinguish normal vs. skewed distributions.
            correlation_threshold: Pearson correlation threshold for dropping collinear features.
            cardinality_threshold: The maximum category threshold for One-Hot encoding.
            noise_threshold: Unique value ratio threshold to classify high-cardinality noise.
        """
        self.target_column = target_column
        self.isolate_target = isolate_target
        self.skewness_threshold = skewness_threshold
        self.correlation_threshold = correlation_threshold
        self.cardinality_threshold = cardinality_threshold
        self.noise_threshold = noise_threshold

        # Attributes learned during fit
        self.numerical_cols_: List[str] = []
        self.categorical_cols_: List[str] = []
        self.dropped_noise_cols_: List[str] = []
        self.impute_values_: Dict[str, Union[float, str]] = {}
        self.log_transformed_cols_: List[str] = []
        self.scalers_: Dict[str, Union[StandardScaler, RobustScaler]] = {}
        self.frequency_maps_: Dict[str, Dict[str, float]] = {}
        self.one_hot_categories_: Dict[str, List[str]] = {}
        self.dropped_collinear_cols_: List[str] = []
        self.missing_ratios_: Dict[str, float] = {}
        self.is_fitted_ = False

    def _validate_input(self, X: pd.DataFrame) -> None:
        """Validates that input is a non-empty Pandas DataFrame."""
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")
        if X.empty:
            raise ValueError("Input DataFrame is empty.")

    def _fit_clean(self, df: pd.DataFrame) -> None:
        """Fits parameters for dynamic column type classification, noise filtering, and imputation."""
        df_working = df.copy()

        # Isolate target column from parameter fitting
        if self.isolate_target and self.target_column:
            if self.target_column in df_working.columns:
                df_working = df_working.drop(columns=[self.target_column])
            else:
                raise KeyError(f"Target column '{self.target_column}' not found in DataFrame columns.")

        # Calculate pre-imputation missing ratios for all columns based on original messy data
        self.missing_ratios_ = (df_working.isnull().sum() / len(df_working)).to_dict()

        # Scan ALL columns globally for high-cardinality structural noise (e.g., completely unique keys, IDs)
        self.dropped_noise_cols_ = []
        remaining_features = []
        
        for col in df_working.columns:
            non_null_count = df_working[col].notnull().sum()
            if non_null_count > 0:
                unique_ratio = df_working[col].nunique() / non_null_count
                if unique_ratio > self.noise_threshold:
                    self.dropped_noise_cols_.append(col)
                    logger.info("Column '%s' flagged as high-cardinality noise (ratio: %.2f) and scheduled for ML drop.", col, unique_ratio)
                else:
                    remaining_features.append(col)
            else:
                self.dropped_noise_cols_.append(col)
                logger.info("Column '%s' is completely null and dropped.", col)

        # Segregate valid features cleanly into structural arrays
        df_remaining = df_working[remaining_features]
        self.numerical_cols_ = df_remaining.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols_ = df_remaining.select_dtypes(exclude=[np.number]).columns.tolist()

        # Compute Imputation Mappings
        self.impute_values_ = {}
        for col in self.numerical_cols_:
            if df_working[col].notnull().sum() == 0:
                self.impute_values_[col] = 0.0
                continue
            
            skewness = df_working[col].skew()
            if pd.isna(skewness):
                skewness = 0.0

            if abs(skewness) <= self.skewness_threshold:
                self.impute_values_[col] = float(df_working[col].mean())
            else:
                self.impute_values_[col] = float(df_working[col].median())

        for col in self.categorical_cols_:
            mode_series = df_working[col].mode()
            if not mode_series.empty:
                self.impute_values_[col] = str(mode_series.iloc[0])
            else:
                self.impute_values_[col] = "UNKNOWN"

    def _fit_mathematical(self, df_clean: pd.DataFrame) -> None:
        """Fits parameters for log-transformations, scaling, categorical encoding, and Pearson correlation pruning."""
        temp_df = df_clean.copy()

        # Strip high-cardinality noise (like customer_id) immediately from math space
        if hasattr(self, 'dropped_noise_cols_') and self.dropped_noise_cols_:
            temp_df = temp_df.drop(columns=[c for c in self.dropped_noise_cols_ if c in temp_df.columns], errors="ignore")

        # Preserve target sequence mapping for correlation importance evaluations
        target_series = None
        if self.target_column in temp_df.columns:
            target_series = temp_df[self.target_column].copy()

        if self.isolate_target and self.target_column:
            if self.target_column in temp_df.columns:
                temp_df = temp_df.drop(columns=[self.target_column])

        # 1. Log-Transformation configuration layer
        self.log_transformed_cols_ = []
        working_numerical_cols = [c for c in self.numerical_cols_ if c in temp_df.columns]
        
        for col in list(working_numerical_cols):
            if col in temp_df.columns:
                skewness = temp_df[col].skew()
                if pd.isna(skewness):
                    skewness = 0.0
                    
                if skewness > 1.0 and (temp_df[col] >= 0).all():
                    self.log_transformed_cols_.append(col)
                    log_col_name = f"{col}_log"
                    temp_df[log_col_name] = np.log1p(temp_df[col])
                    temp_df = temp_df.drop(columns=[col])
                    
                    working_numerical_cols.remove(col)
                    working_numerical_cols.append(log_col_name)

        # 2. Mathematical Scaling configuration layer
        self.scalers_ = {}
        for col in working_numerical_cols:
            if col in temp_df.columns:
                skewness = temp_df[col].skew()
                if pd.isna(skewness):
                    skewness = 0.0
                    
                if abs(skewness) > self.skewness_threshold:
                    scaler = RobustScaler()
                    scaler.fit(temp_df[[col]])
                    self.scalers_[col] = scaler
                else:
                    scaler = StandardScaler()
                    scaler.fit(temp_df[[col]])
                    self.scalers_[col] = scaler

        # Transform temp matrix into scaled space for accurate correlation matrices
        for col in working_numerical_cols:
            if col in temp_df.columns:
                temp_df[col] = self.scalers_[col].transform(temp_df[[col]])

        # 3. Categorical Dummy Variable Expansion configuration layer
        self.one_hot_categories_ = {}
        self.frequency_maps_ = {}
        
        working_categorical_cols = [c for c in self.categorical_cols_ if c in temp_df.columns]
        for col in working_categorical_cols:
            if col in temp_df.columns:
                unique_vals = [str(x) for x in temp_df[col].unique()]
                unique_vals.sort()
                cardinality = len(unique_vals)
                
                if cardinality <= self.cardinality_threshold:
                    self.one_hot_categories_[col] = unique_vals
                    categories_to_encode = unique_vals[1:] if len(unique_vals) > 1 else []
                    for cat in categories_to_encode:
                        temp_df[f"{col}_{cat}"] = (temp_df[col].astype(str) == str(cat)).astype(float)
                    temp_df = temp_df.drop(columns=[col])
                else:
                    freq_map = temp_df[col].value_counts(normalize=True).to_dict()
                    self.frequency_maps_[col] = freq_map
                    temp_df[col] = temp_df[col].astype(str).map(freq_map).fillna(0.0)

        # 4. Target-Aware Pearson Correlation Pruning layer
        self.dropped_collinear_cols_ = []
        all_encoded_features = [col for col in temp_df.columns if col != self.target_column]
        
        if len(all_encoded_features) > 1:
            target_is_numeric = target_series is not None and pd.api.types.is_numeric_dtype(target_series)
            
            # Map absolute target correlation weights for clean tie-breaking decisions
            target_corr_weights = {}
            if target_is_numeric:
                for col in all_encoded_features:
                    correlation_to_target = temp_df[col].corr(target_series)
                    target_corr_weights[col] = abs(correlation_to_target) if pd.notna(correlation_to_target) else 0.0
            
            corr_matrix = temp_df[all_encoded_features].corr(method="pearson").abs()
            dropped_set = set()
            
            for i in range(len(all_encoded_features)):
                col1 = all_encoded_features[i]
                if col1 in dropped_set:
                    continue
                for j in range(i + 1, len(all_encoded_features)):
                    col2 = all_encoded_features[j]
                    if col2 in dropped_set:
                        continue
                    
                    corr_val = corr_matrix.loc[col1, col2]
                    if pd.notna(corr_val) and corr_val > self.correlation_threshold:
                        orig_col1 = col1.replace("_log", "")
                        orig_col2 = col2.replace("_log", "")
                        
                        is_target1 = (col1 == self.target_column) or (orig_col1 == self.target_column)
                        is_target2 = (col2 == self.target_column) or (orig_col2 == self.target_column)
                        
                        if is_target1 and is_target2:
                            continue
                        elif is_target1:
                            col_to_drop = col2
                        elif is_target2:
                            col_to_drop = col1
                        else:
                            # Retrieve predictive strength relative to the target
                            target_strength1 = target_corr_weights.get(col1, 0.0)
                            target_strength2 = target_corr_weights.get(col2, 0.0)
                            
                            if abs(target_strength1 - target_strength2) > 1e-4:
                                # Retain the feature with higher target relationship variance
                                if target_strength1 < target_strength2:
                                    col_to_drop = col1
                                else:
                                    col_to_drop = col2
                            else:
                                # Fallback to completion or ordering rules
                                ratio1 = self.missing_ratios_.get(orig_col1, 0.0)
                                ratio2 = self.missing_ratios_.get(orig_col2, 0.0)
                                if ratio1 > ratio2:
                                    col_to_drop = col1
                                elif ratio2 > ratio1:
                                    col_to_drop = col2
                                else:
                                    col_to_drop = col1 if col1 > col2 else col2
                        
                        dropped_set.add(col_to_drop)
                        self.dropped_collinear_cols_.append(col_to_drop)

    def fit(self, X: pd.DataFrame, y: Optional[Union[pd.Series, pd.DataFrame]] = None) -> "AutoDataPrepPipeline":
        """Fits all steps of the preprocessing pipeline sequentially."""
        self._validate_input(X)
        X_clean = X.copy().drop_duplicates()
        self._fit_clean(X_clean)
        cleaned_df = self.transform_clean(X_clean)
        self._fit_mathematical(cleaned_df)
        self.is_fitted_ = True
        return self

    def transform_clean(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforms input DataFrame by handling null distributions. Preserves original columns for BI reporting."""
        if not self.is_fitted_ and not self.impute_values_:
            raise ValueError("AutoDataPrepPipeline instance must be fitted before transforming.")
        
        self._validate_input(X)
        df = X.copy()

        target_series = None
        has_isolated_target = False
        if self.isolate_target and self.target_column:
            if self.target_column in df.columns:
                target_series = df[self.target_column]
                df = df.drop(columns=[self.target_column])
                has_isolated_target = True

        # Impute missing spots safely using profiled parameters
        for col, val in self.impute_values_.items():
            if col in df.columns:
                df[col] = df[col].fillna(val)

        if has_isolated_target and target_series is not None:
            df[self.target_column] = target_series.values

        return df

    def transform_model_ready(self, X_clean: pd.DataFrame) -> pd.DataFrame:
        """Transforms intermediate cleaned arrays into scaled, encoded, pruned math matrices."""
        if not self.is_fitted_ and not self.scalers_:
            raise ValueError("AutoDataPrepPipeline instance must be fitted before transforming.")
        
        self._validate_input(X_clean)
        df = X_clean.copy()

        target_series = None
        has_isolated_target = False
        if self.isolate_target and self.target_column:
            if self.target_column in df.columns:
                target_series = df[self.target_column]
                df = df.drop(columns=[self.target_column])
                has_isolated_target = True

        # Drop structural noise features immediately inside the ML calculation zone
        if hasattr(self, 'dropped_noise_cols_') and self.dropped_noise_cols_:
            df = df.drop(columns=[c for c in self.dropped_noise_cols_ if c in df.columns], errors="ignore")

        # 1. Log Transformations
        for col in self.log_transformed_cols_:
            if col in df.columns:
                log_col_name = f"{col}_log"
                df[log_col_name] = np.log1p(df[col])
                df = df.drop(columns=[col])

        # 2. Feature Variance Scaling
        for col, scaler in self.scalers_.items():
            if col in df.columns:
                df[col] = scaler.transform(df[[col]])

        # 3. Categorical Encoding (One-Hot & Frequency mapping vectorization)
        for col, categories in self.one_hot_categories_.items():
            if col in df.columns:
                categories_to_encode = categories[1:] if len(categories) > 1 else []
                for cat in categories_to_encode:
                    new_col_name = f"{col}_{cat}"
                    df[new_col_name] = (df[col].astype(str) == str(cat)).astype(float)
                df = df.drop(columns=[col])

        for col, freq_map in self.frequency_maps_.items():
            if col in df.columns:
                df[col] = df[col].astype(str).map(freq_map).fillna(0.0)

        # 4. Correlation Redundancy Feature Pruning
        df = df.drop(columns=[c for c in self.dropped_collinear_cols_ if c in df.columns], errors="ignore")

        if has_isolated_target and target_series is not None:
            df[self.target_column] = target_series.values

        return df

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforms input dataset through both cleaning and mathematical stages."""
        cleaned_df = self.transform_clean(X)
        return self.transform_model_ready(cleaned_df)

    def fit_transform(self, X: pd.DataFrame, y: Optional[Union[pd.Series, pd.DataFrame]] = None) -> pd.DataFrame:
        """Fits the pipeline on the input dataset and returns the fully transformed DataFrame."""
        self._validate_input(X)
        X_clean = X.drop_duplicates()
        return self.fit(X_clean).transform(X_clean)

    def fit_transform_dual(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Dual-output lifecycle workflow for entire complete blocks."""
        self._validate_input(df)
        df_clean = df.drop_duplicates()
        
        self._fit_clean(df_clean)
        cleaned_df = self.transform_clean(df_clean)
        
        self._fit_mathematical(cleaned_df)
        model_ready_df = self.transform_model_ready(cleaned_df)
        
        self.is_fitted_ = True
        return cleaned_df, model_ready_df