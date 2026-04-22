import re
import os
import json

"""
Merchant Database for Deterministic Categorization
This handles OBVIOUS transactions that don't need LLM analysis.
"""

class MerchantDatabase:
    """
    Keyword-based categorization for obvious merchants
    NO LLM CALLS - pure pattern matching
    """
    
    def __init__(self):
        print("🏪 Initializing Merchant Database...")
        self.user_rules_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'data',
            'user_category_rules.json'
        )
        
        # Extensive keyword database (professor emphasized this)
        self.merchant_keywords = {
            'food_dining': [
                # Coffee & Fast Food
                'starbucks', 'dunkin', 'dunkin donuts', 'coffee', 'cafe',
                'mcdonalds', 'burger king', 'subway', 'kfc', 'taco bell',
                'chipotle', 'panera', 'panda express', 'pizza hut', 'dominos',
                
                # Restaurants
                'restaurant', 'bistro', 'grill', 'kitchen', 'diner', 'eatery',
                'food truck', 'catering', 'bakery',
                
                # Food Delivery
                'uber eats', 'doordash', 'grubhub', 'postmates', 'food delivery',

                # Spanish keywords
                'restaurante', 'cafeteria', 'cafe', 'comida', 'almuerzo', 'desayuno',
                'cena', 'sandwicheria', 'pizzeria', 'delivery'
            ],
            
            'groceries': [
                'walmart', 'target', 'costco', 'sams club', 'sam\'s club',
                'kroger', 'safeway', 'publix', 'wegmans', 'giant', 'stop shop',
                'whole foods', 'trader joe', 'aldi', 'food lion', 'harris teeter',
                'market', 'grocery', 'supermarket', 'supercenter', 'food store',
                'supermercado', 'almacen', 'minimarket', 'feria'
            ],
            
            'transportation': [
                # Gas Stations
                'shell', 'exxon', 'chevron', 'bp', 'mobil', 'citgo', 'arco',
                'gas station', 'fuel', 'gasoline', 'petrol',
                
                # Rideshare & Transit
                'uber', 'lyft', 'taxi', 'cab', 'rideshare',
                'metro', 'mta', 'transit', 'bus', 'train', 'subway',
                'parking', 'garage', 'meter',
                
                # Travel
                'airline', 'airport', 'flight', 'car rental', 'hertz', 'enterprise',
                'bencina', 'peaje', 'autopista', 'combustible', 'estacion de servicio', 'metro de'
            ],
            
            'shopping': [
                'amazon', 'ebay', 'etsy', 'best buy', 'apple store', 'microsoft',
                'home depot', 'lowes', 'macys', 'kohls', 'tj maxx', 'marshalls',
                'ross', 'old navy', 'gap', 'nike', 'adidas', 'mall', 'outlet',
                'tienda', 'retail', 'falabella', 'paris', 'ripley', 'mercadolibre'
            ],
            
            'bills_utilities': [
                'electric', 'power', 'energy', 'utility', 'water', 'sewer',
                'gas bill', 'internet', 'cable', 'phone', 'wireless',
                'verizon', 'att', 'at&t', 'comcast', 'spectrum', 'xfinity',
                'municipal', 'city of', 'county of',
                'luz', 'agua', 'gas', 'telefono', 'movil', 'celular', 'entel', 'movistar', 'wom', 'vtr', 'claro'
            ],
            
            'entertainment': [
                'netflix', 'spotify', 'hulu', 'disney', 'amazon prime',
                'apple music', 'youtube', 'gaming', 'steam', 'playstation',
                'xbox', 'nintendo', 'movie', 'theater', 'cinema', 'concert'
            ],
            
            'healthcare': [
                'cvs', 'walgreens', 'rite aid', 'pharmacy', 'medical',
                'doctor', 'dentist', 'hospital', 'clinic', 'health',
                'farmacia', 'medico', 'dentista', 'clinica', 'isapre', 'fonasa'
            ],
            
            'atm_cash': [
                'atm', 'withdrawal', 'cash advance', 'cash back', 'cashout'
            ],
            
            'income': [
                'direct deposit', 'salary', 'payroll', 'interest', 'dividend',
                'refund', 'tax refund', 'deposit', 'credit',
                'abono', 'sueldo', 'nomina', 'transferencia recibida', 'devolucion'
            ],
            
            'fees': [
                'fee', 'charge', 'penalty', 'overdraft', 'maintenance',
                'service charge', 'foreign', 'atm fee',
                'comision', 'mantencion', 'cargo por servicio', 'sobregiro'
            ]
        }
        
        categories_count = sum(len(keywords) for keywords in self.merchant_keywords.values())
        print(f"   📚 Loaded {len(self.merchant_keywords)} categories")
        print(f"   🔑 Total keywords: {categories_count}")

        # User rules are exact/normalized description overrides learned from manual review.
        self.user_category_overrides = self._load_user_category_overrides()
        print(f"   🧠 User category rules: {len(self.user_category_overrides)}")
    
    def categorize_transaction(self, description: str) -> tuple:
        """
        Categorize transaction based on merchant description
        Returns: (category, confidence) or ('uncategorized', 0.0)
        """
        desc_lower = description.lower().strip()
        
        # Remove common prefixes/suffixes that don't help categorization
        desc_clean = self._clean_description_for_matching(desc_lower)

        # First priority: user-learned overrides from manual review.
        if desc_clean in self.user_category_overrides:
            return self.user_category_overrides[desc_clean], 0.99
        
        # Check each category's keywords
        for category, keywords in self.merchant_keywords.items():
            for keyword in keywords:
                if keyword in desc_clean:
                    confidence = self._calculate_confidence(keyword, desc_clean)
                    return category, confidence
        
        # No match found - will need LLM
        return 'uncategorized', 0.0

    def save_user_category_rule(self, description: str, category: str) -> bool:
        """
        Persist a user override so future statements are categorized deterministically.
        """
        if not description or not category:
            return False

        normalized_desc = self._clean_description_for_matching(description.lower().strip())
        if not normalized_desc:
            return False

        self.user_category_overrides[normalized_desc] = category
        return self._persist_user_category_overrides()

    def _load_user_category_overrides(self) -> dict:
        """Load persisted manual categorization rules from disk."""
        if not os.path.exists(self.user_rules_path):
            return {}

        try:
            with open(self.user_rules_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            return {}

        return {}

    def _persist_user_category_overrides(self) -> bool:
        """Save manual categorization rules to disk."""
        try:
            os.makedirs(os.path.dirname(self.user_rules_path), exist_ok=True)
            with open(self.user_rules_path, 'w', encoding='utf-8') as f:
                json.dump(self.user_category_overrides, f, indent=2, sort_keys=True)
            return True
        except Exception:
            return False
    
    def _clean_description_for_matching(self, description: str) -> str:
        """Clean description to improve keyword matching"""
        # Remove common noise
        desc = re.sub(r'[#*]\w*', '', description)  # Remove reference codes
        desc = re.sub(r'\d{4,}', '', desc)  # Remove long numbers
        desc = re.sub(r'\s+', ' ', desc)    # Normalize spaces
        return desc.strip()
    
    def _calculate_confidence(self, matched_keyword: str, description: str) -> float:
        """Calculate confidence based on keyword match quality"""
        
        # Exact brand matches get high confidence
        if matched_keyword in ['starbucks', 'walmart', 'amazon', 'netflix']:
            return 0.95
        
        # Specific store types get high confidence  
        if matched_keyword in ['gas station', 'grocery', 'pharmacy']:
            return 0.90
        
        # Generic keywords get medium confidence
        if matched_keyword in ['restaurant', 'cafe', 'market']:
            return 0.75
        
        # Default confidence for keyword matches
        return 0.80
    
    def get_statistics(self) -> dict:
        """Return database statistics"""
        return {
            'total_categories': len(self.merchant_keywords),
            'total_keywords': sum(len(keywords) for keywords in self.merchant_keywords.values()),
            'categories': list(self.merchant_keywords.keys())
        }