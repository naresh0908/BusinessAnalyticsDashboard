"""
===============================================================================
DATASET GENERATION SCRIPT
===============================================================================
This script generates TWO RELATED DATASETS for the Business Analytics System:

1. SALES TRANSACTIONS (Structured Data - sales_transactions.csv)
   - Contains: Order ID, Customer ID, Product ID, Category, Quantity, 
     Unit Price, Total Amount, Order Date, Region, Payment Method, etc.

2. CUSTOMER REVIEWS (Semi-Structured Data - customer_reviews.csv)
   - Contains: Review ID, Customer ID, Product ID, Rating, Review Text,
     Review Date, Sentiment, Helpful Votes, etc.

RELATIONSHIP: Both datasets share Customer ID and Product ID columns,
allowing them to be MERGED for integrated analysis.
===============================================================================
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# ============================================================================
# CONFIGURATION
# ============================================================================

NUM_CUSTOMERS = 500
NUM_PRODUCTS = 80
NUM_TRANSACTIONS = 5000
NUM_REVIEWS = 2000
DATE_START = datetime(2023, 1, 1)
DATE_END = datetime(2025, 12, 31)

# Product categories and names
CATEGORIES = {
    'Electronics': ['Laptop', 'Smartphone', 'Tablet', 'Headphones', 'Smartwatch',
                    'Wireless Speaker', 'USB-C Hub', 'Power Bank', 'Monitor', 'Keyboard'],
    'Clothing': ['T-Shirt', 'Jeans', 'Jacket', 'Sneakers', 'Dress Shirt',
                 'Hoodie', 'Shorts', 'Formal Shoes', 'Cap', 'Scarf'],
    'Home & Kitchen': ['Coffee Maker', 'Blender', 'Toaster', 'Vacuum Cleaner', 'Air Purifier',
                       'Cookware Set', 'Knife Set', 'Water Bottle', 'Lunch Box', 'Thermos'],
    'Books': ['Fiction Novel', 'Self-Help Book', 'Technical Manual', 'Cookbook', 'Travel Guide',
              'Biography', 'Science Book', 'History Book', 'Art Book', 'Poetry Collection'],
    'Sports': ['Yoga Mat', 'Dumbbells', 'Running Shoes', 'Tennis Racket', 'Basketball',
               'Cycling Helmet', 'Fitness Tracker', 'Resistance Bands', 'Jump Rope', 'Water Bottle'],
    'Beauty': ['Moisturizer', 'Sunscreen', 'Face Wash', 'Shampoo', 'Perfume',
               'Lip Balm', 'Hair Oil', 'Body Lotion', 'Eye Cream', 'Face Mask'],
    'Grocery': ['Organic Honey', 'Green Tea', 'Almonds', 'Olive Oil', 'Dark Chocolate',
                'Quinoa', 'Chia Seeds', 'Protein Bar', 'Oats', 'Peanut Butter'],
    'Toys': ['Board Game', 'Puzzle Set', 'Action Figure', 'LEGO Set', 'Stuffed Animal',
             'Remote Control Car', 'Science Kit', 'Art Supplies', 'Card Game', 'Building Blocks'],
}

REGIONS = ['North', 'South', 'East', 'West', 'Central']
PAYMENT_METHODS = ['Credit Card', 'Debit Card', 'UPI', 'Net Banking', 'Cash on Delivery', 'Wallet']
CITIES = {
    'North': ['Delhi', 'Chandigarh', 'Lucknow', 'Jaipur', 'Amritsar'],
    'South': ['Bangalore', 'Chennai', 'Hyderabad', 'Kochi', 'Mysore'],
    'East': ['Kolkata', 'Bhubaneswar', 'Patna', 'Guwahati', 'Ranchi'],
    'West': ['Mumbai', 'Pune', 'Ahmedabad', 'Goa', 'Surat'],
    'Central': ['Bhopal', 'Indore', 'Nagpur', 'Raipur', 'Jabalpur'],
}

# Review templates for generating realistic review text
POSITIVE_REVIEWS = [
    "Excellent product! Exceeded my expectations. The quality is top-notch and delivery was fast.",
    "Really happy with this purchase. Works perfectly and looks great. Highly recommended!",
    "Amazing value for money. I've been using it for weeks and it's still performing flawlessly.",
    "Best purchase I've made this year. The build quality and design are premium.",
    "Very satisfied with this product. It arrived on time and exactly as described.",
    "Outstanding quality! This is exactly what I was looking for. Five stars!",
    "Great product, great price. Customer service was also very helpful.",
    "Love it! It's become an essential part of my daily routine. So glad I bought it.",
    "Impressive quality at this price point. Definitely worth every penny.",
    "Superb product! My family loves it too. Will definitely buy more from this brand.",
]

NEUTRAL_REVIEWS = [
    "Decent product. Does what it's supposed to do but nothing extraordinary.",
    "It's okay for the price. Not the best quality but functional.",
    "Average product. Some features are good but there's room for improvement.",
    "Works fine but the packaging could have been better. Product is acceptable.",
    "Neither great nor terrible. Gets the job done. Might look for alternatives next time.",
    "Fair product for casual use. Wouldn't recommend for heavy-duty purposes.",
    "It's alright. Matches the description but I expected a bit more.",
    "Satisfactory purchase. The product works but feels a bit cheaply made.",
]

NEGATIVE_REVIEWS = [
    "Very disappointed with the quality. Broke within a week of purchase.",
    "Poor product. Does not match the description at all. Waste of money.",
    "Terrible experience. The product arrived damaged and customer support was unhelpful.",
    "Not worth the price. Quality is really bad and it stopped working quickly.",
    "Regret buying this. The material is cheap and the performance is below average.",
    "Bad product. Would not recommend to anyone. Looking for a refund.",
    "Worst purchase ever. The product is nothing like what was advertised.",
]


def generate_products():
    """Generate a product catalog."""
    products = []
    product_id = 1
    for category, items in CATEGORIES.items():
        for item in items:
            price_ranges = {
                'Electronics': (999, 89999),
                'Clothing': (299, 5999),
                'Home & Kitchen': (499, 14999),
                'Books': (149, 1999),
                'Sports': (199, 9999),
                'Beauty': (99, 4999),
                'Grocery': (49, 1499),
                'Toys': (199, 4999),
            }
            low, high = price_ranges[category]
            price = round(random.uniform(low, high), 2)
            products.append({
                'product_id': f'P{product_id:04d}',
                'product_name': item,
                'category': category,
                'unit_price': price,
                'cost_price': round(price * random.uniform(0.4, 0.75), 2),
            })
            product_id += 1
    return products


def generate_customers():
    """Generate customer data."""
    first_names = ['Aarav', 'Vivaan', 'Aditya', 'Vihaan', 'Arjun', 'Sai', 'Reyansh',
                   'Ayaan', 'Krishna', 'Ishaan', 'Ananya', 'Diya', 'Myra', 'Sara',
                   'Aadhya', 'Isha', 'Kiara', 'Riya', 'Priya', 'Neha', 'Rahul',
                   'Amit', 'Vikram', 'Rohan', 'Karan', 'Pooja', 'Sneha', 'Meera',
                   'Tanvi', 'Kavya', 'Rajesh', 'Sunil', 'Deepak', 'Manoj', 'Suresh',
                   'Preeti', 'Swati', 'Nisha', 'Divya', 'Anjali']
    last_names = ['Sharma', 'Verma', 'Patel', 'Kumar', 'Singh', 'Reddy', 'Nair',
                  'Gupta', 'Joshi', 'Mishra', 'Iyer', 'Rao', 'Pillai', 'Desai',
                  'Mehta', 'Shah', 'Chauhan', 'Yadav', 'Pandey', 'Srivastava']
    
    customers = []
    for i in range(1, NUM_CUSTOMERS + 1):
        region = random.choice(REGIONS)
        city = random.choice(CITIES[region])
        age = random.randint(18, 65)
        gender = random.choice(['Male', 'Female', 'Other'])
        segment = np.random.choice(['Premium', 'Regular', 'Budget'],
                                    p=[0.15, 0.55, 0.30])
        customers.append({
            'customer_id': f'C{i:04d}',
            'customer_name': f'{random.choice(first_names)} {random.choice(last_names)}',
            'age': age,
            'gender': gender,
            'city': city,
            'region': region,
            'customer_segment': segment,
            'registration_date': (DATE_START + timedelta(
                days=random.randint(0, (DATE_END - DATE_START).days)
            )).strftime('%Y-%m-%d'),
        })
    return customers


def generate_transactions(customers, products):
    """Generate sales transaction data."""
    transactions = []
    
    for i in range(1, NUM_TRANSACTIONS + 1):
        customer = random.choice(customers)
        product = random.choice(products)
        order_date = DATE_START + timedelta(
            days=random.randint(0, (DATE_END - DATE_START).days)
        )
        quantity = random.choices(
            [1, 2, 3, 4, 5],
            weights=[0.45, 0.25, 0.15, 0.10, 0.05]
        )[0]
        
        # Add some realistic noise
        discount = random.choice([0, 0, 0, 5, 10, 15, 20, 25])
        unit_price = product['unit_price']
        total_amount = round(unit_price * quantity * (1 - discount / 100), 2)
        
        # Introduce some data quality issues (for cleaning demonstration)
        shipping_cost = round(random.uniform(0, 150), 2) if random.random() > 0.1 else None
        delivery_days = random.randint(1, 14) if random.random() > 0.05 else None
        
        # Some orders might have missing payment methods (dirty data)
        payment = random.choice(PAYMENT_METHODS) if random.random() > 0.03 else None
        
        # Order status
        status_weights = [0.75, 0.10, 0.08, 0.05, 0.02]
        status = random.choices(
            ['Delivered', 'Shipped', 'Processing', 'Cancelled', 'Returned'],
            weights=status_weights
        )[0]
        
        transactions.append({
            'order_id': f'ORD{i:06d}',
            'customer_id': customer['customer_id'],
            'product_id': product['product_id'],
            'product_name': product['product_name'],
            'category': product['category'],
            'quantity': quantity,
            'unit_price': unit_price,
            'discount_percent': discount,
            'total_amount': total_amount,
            'cost_price': product['cost_price'] * quantity,
            'order_date': order_date.strftime('%Y-%m-%d'),
            'region': customer['region'],
            'city': customer['city'],
            'payment_method': payment,
            'shipping_cost': shipping_cost,
            'delivery_days': delivery_days,
            'order_status': status,
            'customer_segment': customer['customer_segment'],
        })
        
    # Add some duplicate rows intentionally (dirty data for cleaning demo)
    num_duplicates = 50
    for _ in range(num_duplicates):
        dup = random.choice(transactions).copy()
        transactions.append(dup)
    
    random.shuffle(transactions)
    return transactions


def generate_reviews(customers, products):
    """Generate customer review data."""
    reviews = []
    
    for i in range(1, NUM_REVIEWS + 1):
        customer = random.choice(customers)
        product = random.choice(products)
        review_date = DATE_START + timedelta(
            days=random.randint(0, (DATE_END - DATE_START).days)
        )
        
        # Generate rating with realistic distribution
        rating = random.choices(
            [1, 2, 3, 4, 5],
            weights=[0.05, 0.10, 0.20, 0.35, 0.30]
        )[0]
        
        # Select review text based on rating
        if rating >= 4:
            review_text = random.choice(POSITIVE_REVIEWS)
        elif rating == 3:
            review_text = random.choice(NEUTRAL_REVIEWS)
        else:
            review_text = random.choice(NEGATIVE_REVIEWS)
        
        # Add some missing reviews (dirty data)
        if random.random() < 0.05:
            review_text = None
        
        # Some ratings might be out of expected range (dirty data)
        noisy_rating = rating
        if random.random() < 0.02:
            noisy_rating = random.choice([0, 6, -1])  # Invalid ratings
        
        helpful_votes = max(0, int(np.random.exponential(3)))
        
        reviews.append({
            'review_id': f'REV{i:06d}',
            'customer_id': customer['customer_id'],
            'product_id': product['product_id'],
            'product_name': product['product_name'],
            'category': product['category'],
            'rating': noisy_rating,
            'review_text': review_text,
            'review_date': review_date.strftime('%Y-%m-%d'),
            'helpful_votes': helpful_votes,
            'verified_purchase': random.choice([True, True, True, False]),
        })
    
    # Add some duplicates (dirty data)
    for _ in range(25):
        dup = random.choice(reviews).copy()
        reviews.append(dup)
    
    random.shuffle(reviews)
    return reviews


def main():
    """Main function to generate and save datasets."""
    print("=" * 60)
    print("  GENERATING BUSINESS ANALYTICS DATASETS")
    print("=" * 60)
    
    # Create data directory
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Generate data
    print("\n[1/4] Generating product catalog...")
    products = generate_products()
    
    print("[2/4] Generating customer database...")
    customers = generate_customers()
    
    print("[3/4] Generating sales transactions...")
    transactions = generate_transactions(customers, products)
    
    print("[4/4] Generating customer reviews...")
    reviews = generate_reviews(customers, products)
    
    # Save to CSV
    df_products = pd.DataFrame(products)
    df_customers = pd.DataFrame(customers)
    df_transactions = pd.DataFrame(transactions)
    df_reviews = pd.DataFrame(reviews)
    
    # Save raw datasets (before cleaning)
    df_transactions.to_csv(os.path.join(data_dir, 'sales_transactions.csv'), index=False)
    df_reviews.to_csv(os.path.join(data_dir, 'customer_reviews.csv'), index=False)
    df_products.to_csv(os.path.join(data_dir, 'product_catalog.csv'), index=False)
    df_customers.to_csv(os.path.join(data_dir, 'customer_database.csv'), index=False)
    
    # Print summary
    print("\n" + "=" * 60)
    print("  DATASETS GENERATED SUCCESSFULLY!")
    print("=" * 60)
    print(f"\n  📁 Saved to: {data_dir}")
    print(f"\n  📊 Dataset 1: sales_transactions.csv")
    print(f"     → {len(df_transactions)} rows, {len(df_transactions.columns)} columns")
    print(f"     → Columns: {', '.join(df_transactions.columns)}")
    print(f"\n  📝 Dataset 2: customer_reviews.csv")
    print(f"     → {len(df_reviews)} rows, {len(df_reviews.columns)} columns")
    print(f"     → Columns: {', '.join(df_reviews.columns)}")
    print(f"\n  📦 Supporting: product_catalog.csv ({len(df_products)} products)")
    print(f"  👥 Supporting: customer_database.csv ({len(df_customers)} customers)")
    print(f"\n  🔗 MERGE KEYS: customer_id, product_id")
    print(f"\n  ⚠️  Intentional data quality issues added for cleaning demo:")
    print(f"     → Missing values in payment_method, shipping_cost, review_text")
    print(f"     → Duplicate rows in both datasets")
    print(f"     → Invalid ratings (0, -1, 6) in reviews")
    print("=" * 60)


if __name__ == '__main__':
    main()
