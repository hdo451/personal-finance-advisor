"""
Main Coordinator: Orchestrates all 3 agents with deterministic routing
====================================================================

This is the main entry point for the entire system.
Uses deterministic if/else routing (NO LLM for routing decisions).

Coordinates:
- Agent 1: Document Processor (0 LLM calls)
- Agent 2: Content Analyzer (1 LLM call)  
- Agent 3: Analysis Generator (0-1 LLM calls)

Total system: ≤2 LLM calls per bank statement
"""

import os
from typing import Dict, Optional
from datetime import datetime
from dotenv import load_dotenv

# Import all agents
from agents.document_processor import DocumentProcessorAgent
from agents.content_analyzer import ContentAnalyzerAgent
from agents.analysis_generator import AnalysisGeneratorAgent
from utils.llm_interface import LLMInterface

class BankStatementAnalyzer:
    """
    Main coordinator using deterministic routing
    Follows exact project requirements for hybrid multi-agent system
    """
    
    def __init__(self, openai_api_key: str):
        """Initialize the complete 3-agent system"""
        
        print("🏦 INITIALIZING BANK STATEMENT ANALYZER")
        print("=" * 50)
        
        # Initialize centralized LLM interface
        self.llm = LLMInterface(openai_api_key)
        
        # Initialize all 3 agents (following project architecture)
        self.agent1 = DocumentProcessorAgent()              # 0 LLM calls
        self.agent2 = ContentAnalyzerAgent(self.llm)        # 1 LLM call
        self.agent3 = AnalysisGeneratorAgent(self.llm)      # 0-1 LLM calls
        
        # System state tracking
        self.session_state = {
            'analyses_performed': 0,
            'total_llm_calls': 0,
            'total_cost': 0.0,
            'last_analysis': None
        }
        
        print(f"✅ All agents initialized successfully")
        print(f"⚠️  WARNING: Educational project only - do not use with real financial data")
        print("")
    
    def analyze_statement(self, pdf_path: str, generate_ai_insights: bool = False) -> Dict:
        """
        Main entry point: Complete bank statement analysis
        
        Uses deterministic routing (if/else logic, not LLM-based)
        
        Args:
            pdf_path: Path to bank statement PDF
            generate_ai_insights: Whether to use LLM for personalized advice
            
        Returns:
            Complete financial analysis results
        """

        # RESET LLM counters for this analysis
        self.llm.reset_counters()
        self.agent1.llm_calls_made = 0
        self.agent2.llm_calls_made = 0  
        self.agent3.llm_calls_made = 0
        
        print(f"🚀 ANALYZING BANK STATEMENT")
        print("=" * 50)
        print(f"📄 File: {os.path.basename(pdf_path)}")
        print(f"🤖 AI insights: {'Enabled' if generate_ai_insights else 'Disabled'}")
        print(f"🕐 Started: {datetime.now().strftime('%H:%M:%S')}")
        print("")
        
        # DETERMINISTIC ROUTING - Step 1: Validation
        validation_result = self._validate_input(pdf_path)
        if not validation_result['valid']:
            return self._create_error_response(validation_result['error'])
        
        try:
            # STEP 1: Agent 1 - Document Processing (0 LLM calls)
            print("1️⃣ AGENT 1: Document Processing")
            print("-" * 30)
            
            doc_result = self.agent1.process(pdf_path)
            
            # Deterministic routing based on Agent 1 results
            if not doc_result['success']:
                return self._create_error_response(
                    f"Document processing failed: {doc_result['error']}",
                    debug_info={'document_processing': doc_result}
                )
            
            transactions = doc_result['transactions']
            
            # Route based on transaction count (deterministic logic)
            if len(transactions) == 0:
                return self._create_error_response(
                    "No transactions found in document. The statement format may need additional parsing rules.",
                    debug_info={
                        'parsing_stats': doc_result.get('parsing_stats', {}),
                        'sample_transaction_lines': doc_result.get('raw_transaction_lines', [])[:10]
                    }
                )
            elif len(transactions) > 500:
                return self._create_error_response("Too many transactions - please split file")
            
            print(f"   ✅ {len(transactions)} transactions processed")
            
            # STEP 2: Agent 2 - Smart Categorization (1 LLM call)
            print(f"\n2️⃣ AGENT 2: Smart Categorization")
            print("-" * 30)
            
            categorized_transactions = self.agent2.process(transactions)
            
            print(f"   ✅ All transactions categorized")
            
            # STEP 3: Agent 3 - Analysis Generation (0-1 LLM calls)
            print(f"\n3️⃣ AGENT 3: Analysis Generation")
            print("-" * 30)
            
            final_analysis = self.agent3.process(
                categorized_transactions, 
                generate_ai_insights=generate_ai_insights
            )
            
            # Calculate final system metrics
            total_llm_calls = (self.agent1.llm_calls_made + 
                             self.agent2.llm_calls_made + 
                             self.agent3.llm_calls_made)
            
            # Update session state
            self._update_session_state(total_llm_calls)
            
            # Create comprehensive result
            result = {
                'success': True,
                'analysis': final_analysis,
                'transactions': categorized_transactions,
                'document_debug': {
                    'parsing_stats': doc_result.get('parsing_stats', {}),
                    'sample_transaction_lines': doc_result.get('raw_transaction_lines', [])[:10]
                },
                'system_metrics': {
                    'total_llm_calls': total_llm_calls,
                    'estimated_cost': self.llm.total_cost,
                    'processing_time': datetime.now().isoformat(),
                    'agent_breakdown': {
                        'agent1_llm_calls': self.agent1.llm_calls_made,
                        'agent2_llm_calls': self.agent2.llm_calls_made,
                        'agent3_llm_calls': self.agent3.llm_calls_made
                    }
                },
                'session_state': self.session_state.copy()
            }
            
            self._print_final_results(result)
            return result
            
        except Exception as e:
            error_msg = f"System error during analysis: {str(e)}"
            print(f"❌ {error_msg}")
            return self._create_error_response(error_msg)
    
    def _validate_input(self, pdf_path: str) -> Dict:
        """Deterministic input validation"""
        
        # Check file exists
        if not os.path.exists(pdf_path):
            return {'valid': False, 'error': f"File not found: {pdf_path}"}
        
        # Check file extension
        if not pdf_path.lower().endswith('.pdf'):
            return {'valid': False, 'error': "Only PDF files are supported"}
        
        # Check file size (basic validation)
        try:
            file_size = os.path.getsize(pdf_path)
            if file_size > 10 * 1024 * 1024:  # 10MB limit
                return {'valid': False, 'error': "File too large (>10MB)"}
            elif file_size == 0:
                return {'valid': False, 'error': "File is empty"}
        except:
            return {'valid': False, 'error': "Cannot access file"}
        
        return {'valid': True}
    
    def _create_error_response(self, error_message: str, debug_info: Optional[Dict] = None) -> Dict:
        """Standardized error response"""
        response = {
            'success': False,
            'error': error_message,
            'system_metrics': {
                'total_llm_calls': 0,
                'estimated_cost': 0.0,
                'agent_breakdown': {
                    'agent1_llm_calls': 0,
                    'agent2_llm_calls': 0, 
                    'agent3_llm_calls': 0
                }
            }
        }
        if debug_info:
            response['debug_info'] = debug_info
        return response
    
    def _update_session_state(self, llm_calls_used: int):
        """Update session tracking"""
        self.session_state['analyses_performed'] += 1
        self.session_state['total_llm_calls'] += llm_calls_used
        self.session_state['total_cost'] += self.llm.total_cost
        self.session_state['last_analysis'] = datetime.now().isoformat()
    
    def _print_final_results(self, result: Dict):
        """Print beautiful final summary"""
        metrics = result['system_metrics']
        analysis = result['analysis']
        summary = analysis['financial_summary']
        
        print(f"\n🎉 ANALYSIS COMPLETE!")
        print("=" * 50)
        print(f"💰 Total Spent: ${summary['total_spent']:.2f}")
        print(f"💵 Total Income: ${summary['total_income']:.2f}")
        print(f"📊 Net Change: ${summary['net_change']:.2f}")
        print(f"🤖 LLM Calls Used: {metrics['total_llm_calls']}/2")
        print(f"💵 Cost: ${metrics['estimated_cost']:.4f}")
        
        if analysis['category_breakdown']:
            top_category = analysis['category_breakdown'][0]
            print(f"🏆 Top Category: {top_category['category']} ({top_category['percentage']:.1f}%)")
        
        print("=" * 50)
    
    def get_system_status(self) -> Dict:
        """Get current system status and metrics"""
        return {
            'agents_initialized': True,
            'session_state': self.session_state.copy(),
            'current_llm_usage': self.llm.get_metrics(),
            'agent_status': {
                'agent1': self.agent1.get_metrics(),
                'agent2': self.agent2.get_metrics(),
                'agent3': self.agent3.get_metrics()
            }
        }

# Command Line Interface (for non-UI usage)
def main_cli():
    """Command line interface for the system"""
    
    # Load API key
    load_dotenv()
    api_key = os.getenv('OPENAI_API_KEY')
    
    if not api_key:
        print("❌ Please set OPENAI_API_KEY in .env file")
        return
    
    # Initialize system
    analyzer = BankStatementAnalyzer(api_key)
    
    while True:
        print(f"\n{'='*60}")
        print("🏦 BANK STATEMENT ANALYZER - CLI")
        print('='*60)
        print("1. Analyze statement (basic)")
        print("2. Analyze statement (with AI insights)")
        print("3. View system status")
        print("4. Exit")
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == '1':
            pdf_path = input("Enter PDF file path: ").strip()
            result = analyzer.analyze_statement(pdf_path, generate_ai_insights=False)
            
        elif choice == '2':
            pdf_path = input("Enter PDF file path: ").strip()
            result = analyzer.analyze_statement(pdf_path, generate_ai_insights=True)
            
        elif choice == '3':
            status = analyzer.get_system_status()
            print(f"\n📊 SYSTEM STATUS:")
            print(f"   Analyses performed: {status['session_state']['analyses_performed']}")
            print(f"   Total LLM calls: {status['session_state']['total_llm_calls']}")
            print(f"   Total cost: ${status['session_state']['total_cost']:.4f}")
            
        elif choice == '4':
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice")

if __name__ == "__main__":
    main_cli()
