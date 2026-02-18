"""
===============================================================================
DATA CLEANING, TRANSFORMATION & MERGING PIPELINE
===============================================================================
This module demonstrates the complete data preparation process:

1. DATA UNDERSTANDING - Explore raw datasets, check types, distributions
2. DATA CLEANING - Handle missing values, remove duplicates, fix invalid data
3. DATA TRANSFORMATION - Feature engineering, type conversion, normalization
4. DATA INTEGRATION (MERGING) - Merge two datasets on shared keys
5. FEATURE SELECTION - Select relevant features for analytics

DATASETS BEING MERGED:
  - sales_transactions.csv (Structured: order records)
  - customer_reviews.csv   (Semi-structured: customer feedback)
  
MERGE KEYS: customer_id + product_id
===============================================================================
"""

import pandas as pd
import numpy as np
import os
import json
from datetime import datetime


class DataPipeline:
    """End-to-end data pipeline for Business Analytics System."""
    
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.raw_transactions = None
        self.raw_reviews = None
        self.clean_transactions_df = None
        self.clean_reviews_df = None
        self.merged_data = None
        self.cleaning_report = {}
        
    # ==================================================================
    # STEP 1: DATA UNDERSTANDING
    # ==================================================================
    
    def load_raw_data(self):
        """Load raw datasets and perform initial exploration."""
        print("\n" + "=" * 60)
        print("  STEP 1: DATA UNDERSTANDING")
        print("=" * 60)
        
        # Load datasets
        self.raw_transactions = pd.read_csv(
            os.path.join(self.data_dir, 'sales_transactions.csv')
        )
        self.raw_reviews = pd.read_csv(
            os.path.join(self.data_dir, 'customer_reviews.csv')
        )
        
        print(f"\n📊 Dataset 1: Sales Transactions")
        print(f"   Shape: {self.raw_transactions.shape}")
        print(f"   Columns: {list(self.raw_transactions.columns)}")
        print(f"   Date Range: {self.raw_transactions['order_date'].min()} to {self.raw_transactions['order_date'].max()}")
        
        print(f"\n📝 Dataset 2: Customer Reviews")
        print(f"   Shape: {self.raw_reviews.shape}")
        print(f"   Columns: {list(self.raw_reviews.columns)}")
        print(f"   Date Range: {self.raw_reviews['review_date'].min()} to {self.raw_reviews['review_date'].max()}")
        
        # Check for common keys
        common_customers = set(self.raw_transactions['customer_id']) & set(self.raw_reviews['customer_id'])
        common_products = set(self.raw_transactions['product_id']) & set(self.raw_reviews['product_id'])
        print(f"\n🔗 Common Customers: {len(common_customers)}")
        print(f"🔗 Common Products: {len(common_products)}")
        
        return self.raw_transactions.copy(), self.raw_reviews.copy()
    
    def get_data_quality_report(self, df, name):
        """Generate a data quality report for a dataset."""
        report = {
            'name': name,
            'rows': len(df),
            'columns': len(df.columns),
            'duplicates': df.duplicated().sum(),
            'missing_values': df.isnull().sum().to_dict(),
            'total_missing': df.isnull().sum().sum(),
            'missing_percent': round(df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100, 2),
            'dtypes': df.dtypes.astype(str).to_dict(),
        }
        return report
    
    # ==================================================================
    # STEP 2: DATA CLEANING
    # ==================================================================
    
    def clean_transactions(self):
        """Clean the sales transactions dataset."""
        print("\n" + "=" * 60)
        print("  STEP 2a: CLEANING SALES TRANSACTIONS")
        print("=" * 60)
        
        df = self.raw_transactions.copy()
        initial_rows = len(df)
        
        # 1. Remove duplicates
        duplicates_before = df.duplicated().sum()
        df = df.drop_duplicates()
        print(f"\n  ✓ Removed {duplicates_before} duplicate rows")
        
        # 2. Handle missing payment_method
        missing_payment = df['payment_method'].isnull().sum()
        df['payment_method'] = df['payment_method'].fillna('Unknown')
        print(f"  ✓ Filled {missing_payment} missing payment methods with 'Unknown'")
        
        # 3. Handle missing shipping_cost
        missing_shipping = df['shipping_cost'].isnull().sum()
        median_shipping = df['shipping_cost'].median()
        df['shipping_cost'] = df['shipping_cost'].fillna(median_shipping)
        print(f"  ✓ Filled {missing_shipping} missing shipping costs with median ({median_shipping})")
        
        # 4. Handle missing delivery_days
        missing_delivery = df['delivery_days'].isnull().sum()
        mean_delivery = round(df['delivery_days'].mean())
        df['delivery_days'] = df['delivery_days'].fillna(mean_delivery)
        print(f"  ✓ Filled {missing_delivery} missing delivery days with mean ({mean_delivery})")
        
        # 5. Convert dates
        df['order_date'] = pd.to_datetime(df['order_date'])
        print(f"  ✓ Converted order_date to datetime")
        
        # 6. Feature Engineering
        df['profit'] = df['total_amount'] - df['cost_price']
        df['profit_margin'] = round((df['profit'] / df['total_amount']) * 100, 2)
        df['order_month'] = df['order_date'].dt.to_period('M').astype(str)
        df['order_year'] = df['order_date'].dt.year
        df['order_quarter'] = df['order_date'].dt.quarter
        df['day_of_week'] = df['order_date'].dt.day_name()
        print(f"  ✓ Created new features: profit, profit_margin, order_month, order_year, order_quarter, day_of_week")
        
        final_rows = len(df)
        print(f"\n  📊 Rows: {initial_rows} → {final_rows} (removed {initial_rows - final_rows})")
        
        self.clean_transactions_df = df
        self.cleaning_report['transactions'] = {
            'initial_rows': initial_rows,
            'final_rows': final_rows,
            'duplicates_removed': duplicates_before,
            'missing_filled': missing_payment + missing_shipping + missing_delivery,
            'features_added': ['profit', 'profit_margin', 'order_month', 'order_year', 'order_quarter', 'day_of_week'],
        }
        return df
    
    def clean_reviews(self):
        """Clean the customer reviews dataset."""
        print("\n" + "=" * 60)
        print("  STEP 2b: CLEANING CUSTOMER REVIEWS")
        print("=" * 60)
        
        df = self.raw_reviews.copy()
        initial_rows = len(df)
        
        # 1. Remove duplicates
        duplicates_before = df.duplicated().sum()
        df = df.drop_duplicates()
        print(f"\n  ✓ Removed {duplicates_before} duplicate rows")
        
        # 2. Fix invalid ratings (must be 1-5)
        invalid_ratings = ((df['rating'] < 1) | (df['rating'] > 5)).sum()
        df['rating'] = df['rating'].clip(1, 5)
        print(f"  ✓ Fixed {invalid_ratings} invalid ratings (clipped to 1-5 range)")
        
        # 3. Handle missing review text
        missing_text = df['review_text'].isnull().sum()
        df['review_text'] = df['review_text'].fillna('No review provided')
        print(f"  ✓ Filled {missing_text} missing review texts")
        
        # 4. Convert dates
        df['review_date'] = pd.to_datetime(df['review_date'])
        print(f"  ✓ Converted review_date to datetime")
        
        # 5. Sentiment classification based on rating
        def classify_sentiment(rating):
            if rating >= 4:
                return 'Positive'
            elif rating == 3:
                return 'Neutral'
            else:
                return 'Negative'
        
        df['sentiment'] = df['rating'].apply(classify_sentiment)
        print(f"  ✓ Added sentiment classification (Positive/Neutral/Negative)")
        
        # 6. Review length feature
        df['review_length'] = df['review_text'].str.len()
        print(f"  ✓ Added review_length feature")
        
        # 7. Review month
        df['review_month'] = df['review_date'].dt.to_period('M').astype(str)
        df['review_year'] = df['review_date'].dt.year
        print(f"  ✓ Added review_month and review_year features")
        
        final_rows = len(df)
        print(f"\n  📊 Rows: {initial_rows} → {final_rows} (removed {initial_rows - final_rows})")
        
        self.clean_reviews_df = df
        self.cleaning_report['reviews'] = {
            'initial_rows': initial_rows,
            'final_rows': final_rows,
            'duplicates_removed': duplicates_before,
            'invalid_ratings_fixed': int(invalid_ratings),
            'missing_filled': int(missing_text),
            'features_added': ['sentiment', 'review_length', 'review_month', 'review_year'],
        }
        return df
    
    # ==================================================================
    # STEP 3: DATA INTEGRATION (MERGING)
    # ==================================================================
    
    def merge_datasets(self):
        """
        Merge sales transactions with customer reviews.
        
        MERGE STRATEGY:
        - LEFT JOIN on (customer_id, product_id)
        - This keeps all transactions and adds review info where available
        - Creates aggregated review metrics per customer-product pair
        """
        print("\n" + "=" * 60)
        print("  STEP 3: DATA INTEGRATION (MERGING)")
        print("=" * 60)
        
        transactions = self.clean_transactions_df.copy()
        reviews = self.clean_reviews_df.copy()
        
        # Aggregate reviews per customer-product pair
        review_agg = reviews.groupby(['customer_id', 'product_id']).agg(
            avg_rating=('rating', 'mean'),
            num_reviews=('review_id', 'count'),
            latest_sentiment=('sentiment', 'last'),
            total_helpful_votes=('helpful_votes', 'sum'),
            avg_review_length=('review_length', 'mean'),
        ).reset_index()
        
        review_agg['avg_rating'] = review_agg['avg_rating'].round(2)
        review_agg['avg_review_length'] = review_agg['avg_review_length'].round(0)
        
        print(f"\n  📊 Transactions: {len(transactions)} rows")
        print(f"  📝 Aggregated Reviews: {len(review_agg)} customer-product pairs")
        
        # MERGE using LEFT JOIN
        merged = pd.merge(
            transactions,
            review_agg,
            on=['customer_id', 'product_id'],
            how='left',
            indicator=True
        )
        
        # Check merge results
        matched = (merged['_merge'] == 'both').sum()
        left_only = (merged['_merge'] == 'left_only').sum()
        
        print(f"\n  🔗 MERGE RESULTS (LEFT JOIN on customer_id + product_id):")
        print(f"     → Matched rows: {matched}")
        print(f"     → Transactions without reviews: {left_only}")
        print(f"     → Final merged dataset: {len(merged)} rows")
        
        # Fill NaN values for unmatched rows
        merged['avg_rating'] = merged['avg_rating'].fillna(0)
        merged['num_reviews'] = merged['num_reviews'].fillna(0).astype(int)
        merged['latest_sentiment'] = merged['latest_sentiment'].fillna('No Review')
        merged['total_helpful_votes'] = merged['total_helpful_votes'].fillna(0).astype(int)
        merged['avg_review_length'] = merged['avg_review_length'].fillna(0)
        
        # Has review flag
        merged['has_review'] = (merged['_merge'] == 'both').astype(int)
        merged = merged.drop(columns=['_merge'])
        
        # Customer satisfaction score (composite)
        merged['satisfaction_score'] = np.where(
            merged['has_review'] == 1,
            (merged['avg_rating'] / 5 * 0.7 + merged['total_helpful_votes'].clip(0, 10) / 10 * 0.3) * 100,
            50  # neutral default
        ).round(1)
        
        print(f"  ✓ Added satisfaction_score composite metric")
        print(f"\n  📊 Final merged dataset: {merged.shape}")
        print(f"  📊 Columns: {list(merged.columns)}")
        
        self.merged_data = merged
        self.cleaning_report['merge'] = {
            'transactions_rows': len(transactions),
            'review_pairs': len(review_agg),
            'matched': int(matched),
            'unmatched': int(left_only),
            'final_rows': len(merged),
            'final_columns': len(merged.columns),
            'merge_type': 'LEFT JOIN',
            'merge_keys': ['customer_id', 'product_id'],
        }
        
        return merged
    
    # ==================================================================
    # STEP 4: SAVE PROCESSED DATA
    # ==================================================================
    
    def save_processed_data(self):
        """Save cleaned and merged datasets."""
        print("\n" + "=" * 60)
        print("  STEP 4: SAVING PROCESSED DATA")
        print("=" * 60)
        
        processed_dir = os.path.join(self.data_dir, 'processed')
        os.makedirs(processed_dir, exist_ok=True)
        
        self.clean_transactions_df.to_csv(
            os.path.join(processed_dir, 'clean_transactions.csv'), index=False
        )
        self.clean_reviews_df.to_csv(
            os.path.join(processed_dir, 'clean_reviews.csv'), index=False
        )
        self.merged_data.to_csv(
            os.path.join(processed_dir, 'merged_analytics_data.csv'), index=False
        )
        
        # Save cleaning report
        with open(os.path.join(processed_dir, 'cleaning_report.json'), 'w') as f:
            json.dump(self.cleaning_report, f, indent=2, default=str)
        
        print(f"\n  ✓ Saved clean_transactions.csv")
        print(f"  ✓ Saved clean_reviews.csv")
        print(f"  ✓ Saved merged_analytics_data.csv")
        print(f"  ✓ Saved cleaning_report.json")
        print(f"\n  📁 Location: {processed_dir}")
        
    # ==================================================================
    # RUN FULL PIPELINE
    # ==================================================================
    
    def run(self):
        """Execute the complete data pipeline."""
        print("\n" + "=" * 60)
        print("  🚀 RUNNING DATA PIPELINE")
        print("=" * 60)
        
        self.load_raw_data()
        self.clean_transactions()
        self.clean_reviews()
        self.merge_datasets()
        self.save_processed_data()
        
        print("\n" + "=" * 60)
        print("  ✅ PIPELINE COMPLETE!")
        print("=" * 60)
        
        return self.merged_data, self.clean_transactions_df, self.clean_reviews_df


if __name__ == '__main__':
    pipeline = DataPipeline()
    pipeline.run()
