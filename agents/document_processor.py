import pdfplumber
import re
from datetime import datetime
from typing import List, Dict, Optional
from .base_agent import BaseAgent
from utils.merchant_database import MerchantDatabase

class DocumentProcessorAgent(BaseAgent):
    """
    Agent 1: Pure deterministic document processing
    NO LLM CALLS - extracts and parses transaction data
    """

    def __init__(self):
        super().__init__("Document Processor", uses_llm=False)
        self.merchant_db = MerchantDatabase()
        print(f"🏗️ {self.name} initialized - 0 LLM calls")

    def process(self, pdf_path: str) -> Dict:
        """Main entry point: PDF → Structured transaction data"""
        print(f"\n📄 {self.name} processing: {pdf_path}")
        
        try:
            # Step 1: Extract raw text
            raw_text = self._extract_text_from_pdf(pdf_path)
            
            if not raw_text.strip():
                return {'success': False, 'error': 'No text in PDF', 'transactions': []}
            
            print(f"   ✅ Extracted {len(raw_text)} characters")
            
            # Step 2: Find transaction lines
            transaction_lines = self._find_transaction_lines(raw_text)
            
            if not transaction_lines:
                return {
                    'success': True, 
                    'transactions': [],
                    'raw_text': raw_text,
                    'message': 'No transaction lines found - might need different parsing approach'
                }
            
            # NEW Step 3: Parse transaction lines into structured data
            parsed_transactions = self._parse_transaction_lines(transaction_lines)
            
            categorized_transactions = self._apply_basic_categorization(parsed_transactions)

            return {
                'success': True,
                'transactions': categorized_transactions,  # Changed from parsed_transactions
                'total_transactions': len(categorized_transactions),
                'raw_transaction_lines': transaction_lines,
                'parsing_stats': {
                    'lines_found': len(transaction_lines),
                    'successfully_parsed': len(parsed_transactions),
                    'parse_success_rate': len(parsed_transactions) / len(transaction_lines) if transaction_lines else 0,
                    'categorized_deterministically': len([t for t in categorized_transactions if t['category'] != 'uncategorized']),
                    'needs_llm': len([t for t in categorized_transactions if t['category'] == 'uncategorized'])
                }
            }

            
        except Exception as e:
            return {'success': False, 'error': str(e), 'transactions': []}
        
    def _extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Enhanced PDF extraction - handles both text and tables
        """
        full_text = ""
        
        with pdfplumber.open(pdf_path) as pdf:
            print(f"   📖 PDF has {len(pdf.pages)} pages")
            
            for page_num, page in enumerate(pdf.pages, 1):
                print(f"   📄 Processing page {page_num}...")
                
                # Method 1: Try table extraction first (for structured data)
                tables = page.extract_tables()
                if tables:
                    print(f"   📊 Found {len(tables)} tables on page {page_num}")
                    
                    for table_num, table in enumerate(tables, 1):
                        full_text += f"\n--- PAGE {page_num} TABLE {table_num} ---\n"
                        
                        for row_num, row in enumerate(table):
                            if row and any(cell for cell in row if cell and str(cell).strip()):
                                # Join non-empty cells with tabs
                                row_text = '\t'.join(str(cell).strip() if cell else '' for cell in row)
                                full_text += f"ROW_{row_num}: {row_text}\n"
                        
                        full_text += f"--- END TABLE {table_num} ---\n"
                
                # Method 2: Regular text extraction as backup
                page_text = page.extract_text()
                if page_text:
                    full_text += f"\n--- PAGE {page_num} TEXT ---\n"
                    full_text += page_text
                    full_text += f"\n--- END PAGE {page_num} TEXT ---\n"
                
                # Method 3: Character-level extraction for stubborn PDFs
                if not tables and not page_text:
                    print(f"   ⚠️  Trying character-level extraction...")
                    chars = page.chars
                    if chars:
                        full_text += f"\n--- PAGE {page_num} CHARS ---\n"
                        for char in chars[:100]:  # First 100 characters for debugging
                            full_text += char.get('text', '')
                        full_text += f"\n--- END CHARS ---\n"
            
            return full_text
    
    def _find_transaction_lines(self, text: str) -> List[str]:
        """
        Find Lines that actually contain transactions
        Only processes transactions after "TRANSACTION DETAIL" marker to avoid header/footer noise
        """
        lines = text.split('\n')
        
        # Find the marker where actual transactions begin
        transaction_start_idx = 0
        for i, line in enumerate(lines):
            if 'transaction detail' in line.lower() or 'detail de transacciones' in line.lower():
                transaction_start_idx = i
                print(f"   📍 Found transaction block at line {i+1}")
                break
        
        # Process only from that point forward
        lines_to_scan = lines[transaction_start_idx:]
        
        potential_transactions = []
        seen_lines = set()  # Deduplicate transactions

        print(f"   🔍 Scanning {len(lines_to_scan)} lines for transactions...")

        for line_num, line in enumerate(lines_to_scan, transaction_start_idx + 1):
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Skip obvious non-transaction lines
            if self._is_header_or_footer(line):
                continue

            # Check if this looks like a transaction
            if self._looks_like_transaction(line):
                # Deduplicate by storing hash of line
                line_hash = hash(line)
                if line_hash not in seen_lines:
                    seen_lines.add(line_hash)
                    potential_transactions.append(line)
                    print(f"   ✅ Line {line_num}: {line[:50]}...")
                else:
                    print(f"   ⏭️  Line {line_num}: Duplicate, skipping")

        print(f"   📊 Found {len(potential_transactions)} potential transaction lines")
        return potential_transactions
    
    def _is_header_or_footer(self, line: str) -> bool:
        """Skip lines that are obviously NOT transactions"""
        line_lower = line.lower()
        
        # Skip header/footer patterns (not anchored to ^ so they match anywhere)
        skip_patterns = [
            r'(account|statement|balance information|date|description|amount|type)',
            r'(cartola|cuenta|saldo anterior|saldo final|saldo inicial|fecha|descripcion|descripci[oó]n|monto|abono|cargo)',
            r'(page \d+|statement period|issue date)',
            r'(opening balance|previous balance|ending balance|current balance|closing balance)',
            # Note: Interest earned DEPOSITS are kept if they have amount; fees are kept too
            r'(interest paid)',  # Skip only interest PAID (fees/charges)
            r'(note:|total|summary)',
            r'(paid in|paid out|detail|payment type|account holder)',
            r'(date\s+description|description.*type|type.*amount)',  # Table headers
            r'^\s*$',  # Empty lines
            r'^-+$',   # Separator lines like "----------"
            r'^=+$'    # Separator lines like "=========="
        ]
        
        for pattern in skip_patterns:
            if re.search(pattern, line_lower):
                return True
        
        return False
    
    def _looks_like_transaction(self, line: str) -> bool:
        """
        Check if line has the pattern of a real transaction
        Must have: date + description + amount
        """
        # Must have a date pattern bounded by word boundaries to prevent matching routing/phone numbers
        date_pattern = r'\b(\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b'
        has_date = bool(re.search(date_pattern, line))
        
        # Must have an amount pattern (strictly requiring either $ or decimal part)
        amount_pattern = r'[-+]?\$?\s*\d{1,3}(?:[\.,]\d{3})+(?:[\.,]\d{2})?|[-+]?\$\s*\d+(?:[\.,]\d{2})?|[-+]?\s*\d+[\.,]\d{2}'
        has_amount = bool(re.search(amount_pattern, line))
        
        # Must have some description text (not just numbers/symbols)
        has_description = bool(re.search(r'[A-Za-zÀ-ÿ]{3,}', line))
        
        return has_date and has_amount and has_description
    
    def _parse_transaction_lines(self, transaction_lines: List[str]) -> List[Dict]:
        """
        Parse each transaction line into structured data
        """
        print(f"   🔧 Parsing {len(transaction_lines)} transaction lines...")
        
        parsed_transactions = []
        failed_parses = 0
        
        for line_num, line in enumerate(transaction_lines, 1):
            try:
                parsed = self._parse_single_transaction(line)
                if parsed:
                    parsed['transaction_id'] = f"txn_{len(parsed_transactions) + 1}"
                    parsed_transactions.append(parsed)
                    if line_num <= 3:  # Show first 3 for debugging
                        print(f"   ✅ Parsed line {line_num}: {parsed['description'][:30]}... = ${parsed['amount']}")
                else:
                    failed_parses += 1
                    if failed_parses <= 2:  # Show first 2 failures
                        print(f"   ❌ Failed to parse: {line[:50]}...")
                        
            except Exception as e:
                failed_parses += 1
                if failed_parses <= 2:
                    print(f"   ❌ Parse error: {line[:30]}... → {e}")

        success_rate = len(parsed_transactions) / len(transaction_lines) if transaction_lines else 0
        print(f"   📊 Parsing success: {len(parsed_transactions)}/{len(transaction_lines)} ({success_rate:.1%})")

        return parsed_transactions

    def _parse_single_transaction(self, line: str) -> Optional[Dict]:
        """
        Parse one transaction line using multiple regex patterns
        Returns None if no pattern matches
        """
        
        # Define regex patterns for different bank formats
        # Strict amount pattern: requires $ symbol OR explicit decimal/comma (rejects phone numbers, IDs)
        strict_amount = r'(?:[-+]?\$\s*\d{1,3}(?:[\.,]\d{3})+(?:[\.,]\d{2})?|[-+]?\$\s*\d+(?:[\.,]\d{2})?|[-+]?\s*\d+[\.,]\d{2})'
        
        patterns = [
            # Pattern 1: Chase format "02/01/2024 DUNKIN DONUTS 8901 -$4.50 $1,851.92"
            {
                'regex': r'(\d{1,2}/\d{1,2}/\d{4})\s+([^-+$\d]+?)\s+(' + strict_amount + r')\s+\$?([\d,]+\.?\d*)',
                'groups': {'date': 1, 'description': 2, 'amount': 3, 'balance': 4},
                'name': 'Chase Format'
            },
            
            # Pattern 2: Wells Fargo format "03-01-2024 BURGER KING #4521 $8.99 $2,136.68"
            {
                'regex': r'(\d{1,2}-\d{1,2}-\d{4})\s+([^$\d]+?)\s+(' + strict_amount + r')\s+\$?([\d,]+\.?\d*)',
                'groups': {'date': 1, 'description': 2, 'amount': 3, 'balance': 4},
                'name': 'Wells Fargo Format'
            },
            
            # Pattern 3: Generic format "01/02/2024 STARBUCKS STORE 1234 -$5.67 $2,494.33"
            {
                'regex': r'(\d{1,2}/\d{1,2}/\d{4})\s+([A-Za-z][^-+$\d]*?)\s+(' + strict_amount + r')\s+\$?([\d,]+\.?\d*)',
                'groups': {'date': 1, 'description': 2, 'amount': 3, 'balance': 4},
                'name': 'Generic Format'
            },

            # Pattern 4: LatAm format with decimal comma and optional balance
            {
                'regex': r'(\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?)\s+(.+?)\s+((?:[-+]?\$?\s*\d{1,3}(?:[\.,]\d{3})+(?:[\.,]\d{2})?|[-+]?\$\s*\d+(?:[\.,]\d{2})?|[-+]?\s*\d+[\.,]\d{2}))(?:\s+((?:[-+]?\$?\s*\d{1,3}(?:[\.,]\d{3})+(?:[\.,]\d{2})?|[-+]?\$\s*\d+(?:[\.,]\d{2})?|[-+]?\s*\d+[\.,]\d{2})))?$',
                'groups': {'date': 1, 'description': 2, 'amount': 3, 'balance': 4},
                'name': 'LatAm Formats'
            },

            # Pattern 5: ISO date plus free-form text, amount at the end
            {
                'regex': r'(\d{4}[/-]\d{1,2}[/-]\d{1,2})\s+(.+?)\s+((?:[-+]?\$?\s*\d{1,3}(?:[\.,]\d{3})+(?:[\.,]\d{2})?|[-+]?\$\s*\d+(?:[\.,]\d{2})?|[-+]?\s*\d+[\.,]\d{2}))$',
                'groups': {'date': 1, 'description': 2, 'amount': 3},
                'name': 'ISO Date Generic'
            }
        ]
        
        # Try each pattern
        for pattern in patterns:
            match = re.search(pattern['regex'], line)
            if match:
                try:
                    return self._extract_transaction_data(match, pattern, line)
                except Exception as e:
                    continue  # Try next pattern

        # Fallback parser for mixed/unexpected formats
        return self._parse_with_tokenization(line)

    def _parse_with_tokenization(self, line: str) -> Optional[Dict]:
        """Fallback parser that identifies date and numeric tokens and avoids using running balance as amount."""
        date_pattern = r'\b(\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b'
        date_match = re.search(date_pattern, line)
        if not date_match:
            return None

        date_str = date_match.group(1)
        parsed_date = self._parse_date(date_str)
        after_date = line[date_match.end():].strip()

        amount_pattern = r'[-+]?\$?\s*\d{1,3}(?:[\.,]\d{3})+(?:[\.,]\d{2})?|[-+]?\$\s*\d+(?:[\.,]\d{2})?|[-+]?\s*\d+[\.,]\d{2}'
        amount_matches = list(re.finditer(amount_pattern, after_date))
        if not amount_matches:
            return None

        # Heuristic: if there are 2+ numeric tokens, last one is often running balance.
        amount_idx = len(amount_matches) - 1
        if len(amount_matches) >= 2:
            amount_idx = len(amount_matches) - 2

        amount_match = amount_matches[amount_idx]
        amount_str = amount_match.group(0).strip()

        description = after_date[:amount_match.start()].strip()
        if not description:
            description = after_date[:amount_matches[0].start()].strip()
        description = re.sub(r'\s+', ' ', description)
        description = re.sub(r'[#*]\w*', '', description).strip()
        if len(description) < 2:
            return None

        amount = self._parse_amount(amount_str)
        is_debit = self._is_debit_transaction(amount_str, line)

        balance = 0.0
        if amount_idx + 1 < len(amount_matches):
            balance = self._parse_amount(amount_matches[amount_idx + 1].group(0).strip())

        return {
            'date': parsed_date.strftime('%Y-%m-%d'),
            'description': description,
            'amount': abs(amount),
            'is_debit': is_debit,
            'balance': balance,
            'category': 'uncategorized',
            'confidence': 0.0,
            'source': 'deterministic',
            'original_line': line,
            'pattern_used': 'Tokenized Fallback'
        }

    def _extract_transaction_data(self, match, pattern: Dict, original_line: str) -> Dict:
        """Extract structured data from successful regex match"""
        
        groups = match.groups()
        group_map = pattern['groups']
        
        # Extract date
        date_str = groups[group_map['date'] - 1]
        parsed_date = self._parse_date(date_str)
        
        # Extract description (clean it up)
        description = groups[group_map['description'] - 1].strip()
        description = re.sub(r'\s+', ' ', description)  # Remove extra spaces
        description = re.sub(r'[#*]\w*', '', description)  # Remove reference codes
        description = description.strip()
        
        # Extract amount
        amount_str = groups[group_map['amount'] - 1]
        amount = self._parse_amount(amount_str)
        is_debit = self._is_debit_transaction(amount_str, original_line)
        
        # Extract balance (if available)
        balance = 0.0
        if 'balance' in group_map and len(groups) >= group_map['balance']:
            balance_str = groups[group_map['balance'] - 1]
            balance = self._parse_amount(balance_str)
        
        return {
            'date': parsed_date.strftime('%Y-%m-%d'),
            'description': description,
            'amount': abs(amount),  # Always positive, use is_debit flag
            'is_debit': is_debit,
            'balance': balance,
            'category': 'uncategorized',  # Will be set later
            'confidence': 0.0,
            'source': 'deterministic',
            'original_line': original_line,
            'pattern_used': pattern['name']
        }

    # Helper methods for parsing components
    def _parse_date(self, date_str: str) -> datetime:
        """Handle different date formats"""
        from datetime import datetime
        
        date_formats = [
            '%m/%d/%Y',     # 02/01/2024
            '%d/%m/%Y',     # 01/02/2024
            '%m-%d-%Y',     # 03-01-2024  
            '%d-%m-%Y',     # 01-03-2024
            '%d/%m/%y',     # 01/02/24
            '%d-%m-%y',     # 01-02-24
            '%Y-%m-%d',     # 2024-02-01
            '%Y/%m/%d',     # 2024/02/01
            '%d-%b-%Y',     # 01-FEB-2024
        ]
        
        for fmt in date_formats:
            try:
                return datetime.strptime(date_str, fmt)
            except:
                continue

        # Handle short date formats without year as current year
        for fmt in ['%d/%m', '%d-%m', '%m/%d', '%m-%d']:
            try:
                parsed = datetime.strptime(date_str, fmt)
                return parsed.replace(year=datetime.now().year)
            except:
                continue
        
        # Fallback to current date if parsing fails
        print(f"   ⚠️  Couldn't parse date: {date_str}")
        return datetime.now()

    def _parse_amount(self, amount_str: str) -> float:
        """Clean and parse amount string"""
        clean_amount = amount_str.strip()
        clean_amount = clean_amount.replace(' ', '')
        clean_amount = re.sub(r'[$€£]', '', clean_amount)

        # Normalize accounting negatives like (1.234,56)
        negative = False
        if clean_amount.startswith('(') and clean_amount.endswith(')'):
            negative = True
            clean_amount = clean_amount[1:-1]

        # Determine decimal separator by the last separator occurrence
        if ',' in clean_amount and '.' in clean_amount:
            if clean_amount.rfind(',') > clean_amount.rfind('.'):
                # 1.234,56 -> decimal comma
                clean_amount = clean_amount.replace('.', '').replace(',', '.')
            else:
                # 1,234.56 -> decimal dot
                clean_amount = clean_amount.replace(',', '')
        elif ',' in clean_amount:
            # If there are exactly 2 decimal digits after comma, treat as decimal separator
            if re.search(r',\d{2}$', clean_amount):
                clean_amount = clean_amount.replace('.', '').replace(',', '.')
            else:
                clean_amount = clean_amount.replace(',', '')
        else:
            # Dot-only numbers: keep decimal dot, remove thousands commas if any
            clean_amount = clean_amount.replace(',', '')

        try:
            value = float(clean_amount)
            return -value if negative else value
        except:
            print(f"   ⚠️  Couldn't parse amount: {amount_str}")
            return 0.0

    def _is_debit_transaction(self, amount_str: str, line: str) -> bool:
        """Determine if transaction is debit (money going out)"""
        
        # Check for explicit negative signs
        if amount_str.startswith('-') or amount_str.startswith('+'):
            return amount_str.startswith('-')

        # Common banking indicators in line or amount
        if re.search(r'\b(db|dr|debit|cargo|compra|egreso|withdrawal|payment|fee|charge)\b', line.lower()):
            return True
        if re.search(r'\b(cr|abono|credito|credit|deposit|salary|refund|income)\b', line.lower()):
            return False
        
        # Check for debit indicators in the line
        debit_keywords = ['withdrawal', 'purchase', 'payment', 'fee', 'charge', 'cargo', 'compra', 'egreso']
        line_lower = line.lower()
        
        for keyword in debit_keywords:
            if keyword in line_lower:
                return True
        
        # Check for credit indicators
        credit_keywords = ['deposit', 'interest', 'refund', 'credit', 'salary', 'abono', 'credito', 'ingreso']
        for keyword in credit_keywords:
            if keyword in line_lower:
                return False
        
        # Default to debit for most transactions
        return True
    
    def _apply_basic_categorization(self, transactions: List[Dict]) -> List[Dict]:
        """
        Step 4: Apply deterministic categorization using merchant database
        NO LLM CALLS - pure keyword matching
        """
        print(f"   🏷️ Applying basic categorization to {len(transactions)} transactions...")
        
        categorized_count = 0
        
        for txn in transactions:
            category, confidence = self.merchant_db.categorize_transaction(txn['description'])
            
            if category != 'uncategorized':
                txn['category'] = category
                txn['confidence'] = confidence
                txn['source'] = 'deterministic'
                categorized_count += 1
            # else: remains 'uncategorized' for Agent 2
        
        categorization_rate = categorized_count / len(transactions) if transactions else 0
        print(f"   📊 Deterministic categorization: {categorized_count}/{len(transactions)} ({categorization_rate:.1%})")
        print(f"   🤖 Will need LLM for: {len(transactions) - categorized_count} transactions")
        
        return transactions