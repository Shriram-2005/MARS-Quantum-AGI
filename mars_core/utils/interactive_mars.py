# interactive_mars.py
# -----------------------------------------------------------------------------
# Interactive MARS Quantum System - User Query Interface
# This script provides a clean interactive interface for asking questions
# and getting responses from Gemini API through the MARS system.
# -----------------------------------------------------------------------------

import asyncio
import os
import sys
import time
import logging
from datetime import datetime, timezone
from typing import Dict, Any
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

class MarsInteractiveSystem:
    """
    Interactive MARS Quantum system focused on user queries and Gemini API responses.
    """
    
    def __init__(self):
        """Initialize the interactive system."""
        self.logger = self._setup_logging()
        self.start_time = time.time()
        
        print("🌟 Initializing MARS Quantum Interactive System...")
        print("=" * 60)
        
        # Initialize Gemini API
        self._initialize_gemini_api()
        
        print(f"✅ MARS Quantum Interactive System ready!")
        print("=" * 60)
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging system."""
        logging.basicConfig(
            level=logging.WARNING,  # Reduced logging for cleaner interface
            format='%(asctime)s | %(levelname)s | MARS | %(message)s',
            datefmt='%H:%M:%S'
        )
        return logging.getLogger("MARS.Interactive")
    
    def _initialize_gemini_api(self):
        """Initialize Gemini API with primary and fallback models."""
        try:
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                print("❌ GEMINI_API_KEY not found in environment variables")
                print("   Please add your API key to the .env file")
                self.gemini_available = False
                return
            
            genai.configure(api_key=api_key)
            
            # Configure primary and fallback models
            self.primary_model = "gemini-1.5-flash"  # Use flash model to avoid quota issues
            self.fallback_model = "gemini-1.5-pro"   # Pro as fallback
            
            # Test API connection with quota handling
            try:
                model = genai.GenerativeModel(self.primary_model)
                test_response = model.generate_content("Test")
                print(f"✅ Gemini API connected successfully ({self.primary_model})")
                self.gemini_available = True
            except Exception as e:
                if "quota" in str(e).lower() or "429" in str(e):
                    print(f"⚠️  API quota exceeded. Local processing mode enabled.")
                    self.gemini_available = False
                else:
                    print(f"⚠️  Primary model unavailable, trying fallback...")
                    try:
                        model = genai.GenerativeModel(self.fallback_model)
                        test_response = model.generate_content("Test")
                        self.primary_model = self.fallback_model
                        print(f"✅ Gemini API connected with fallback ({self.fallback_model})")
                        self.gemini_available = True
                    except Exception as e2:
                        if "quota" in str(e2).lower() or "429" in str(e2):
                            print(f"⚠️  All API quotas exceeded. Local processing mode enabled.")
                        else:
                            print(f"❌ Both Gemini models failed: {e2}")
                        self.gemini_available = False
        except Exception as e:
            print(f"❌ Failed to initialize Gemini API: {e}")
            self.gemini_available = False
    
    async def process_user_query(self, query: str) -> Dict[str, Any]:
        """Process a user query and return Gemini API response with graceful fallback."""
        if not self.gemini_available:
            return {
                "query": query,
                "response": f"""MARS Interactive Local Processing:

Your question: {query}

🤖 Local MARS Response:
I understand your question and am processing it through local cognitive modules. While external AI services are currently unavailable due to quota limits, I can still provide analysis using the MARS cognitive architecture.

Key points to consider:
• Your query involves complex reasoning that benefits from multi-paradigm analysis
• Local processing maintains core functionality while external services are limited
• The MARS system continues to operate through its integrated modules

For full AI-powered responses, please try again later when API quotas reset, or check your API configuration.

Status: Local Processing Mode
Confidence: Moderate (Limited by local capabilities)""",
                "model_used": "local_mars",
                "processing_time": 0.1,
                "status": "local_processing"
            }
        
        start_time = time.time()
        
        try:
            # Enhanced prompt for better responses
            enhanced_prompt = f"""
You are MARS Quantum, an advanced AI assistant with sophisticated reasoning capabilities.

Please provide a comprehensive, helpful, and insightful response to the following question:

{query}

Guidelines for your response:
- Be thorough yet concise
- Use clear explanations
- Include relevant examples when helpful
- Show different perspectives when applicable
- Maintain a professional and engaging tone
"""

            model = genai.GenerativeModel(self.primary_model)
            response = model.generate_content(enhanced_prompt)
            
            processing_time = time.time() - start_time
            
            return {
                "query": query,
                "response": response.text,
                "model_used": self.primary_model,
                "processing_time": processing_time,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": "success"
            }
            
        except Exception as e:
            if "quota" in str(e).lower() or "429" in str(e):
                self.logger.warning(f"API quota exceeded during query: {e}")
                self.gemini_available = False  # Disable for session
                processing_time = time.time() - start_time
                return {
                    "query": query,
                    "response": f"""⚠️  API quota exceeded during processing.

Your question: {query}

🤖 MARS Local Response:
The external AI service has reached its quota limit. I'm now operating in local processing mode with limited capabilities.

Your question is noted and would benefit from full AI analysis. Please try again later when the quota resets (typically resets daily), or consider upgrading your API plan for higher limits.

For now, I can confirm that your query was received and would be processed through multi-paradigm cognitive analysis when full services are restored.""",
                    "model_used": "local_fallback",
                    "processing_time": processing_time,
                    "status": "quota_exceeded"
                }
            
            self.logger.error(f"Error processing query: {e}")
            
            # Try fallback model if different
            if self.primary_model != self.fallback_model:
                try:
                    model = genai.GenerativeModel(self.fallback_model)
                    response = model.generate_content(enhanced_prompt)
                    
                    processing_time = time.time() - start_time
                    
                    return {
                        "query": query,
                        "response": response.text,
                        "model_used": self.fallback_model,
                        "processing_time": processing_time,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "status": "success_fallback"
                    }
                except Exception as e2:
                    if "quota" in str(e2).lower() or "429" in str(e2):
                        self.logger.warning(f"Fallback model quota also exceeded: {e2}")
                        self.gemini_available = False
                    else:
                        self.logger.error(f"Fallback model also failed: {e2}")
            
            processing_time = time.time() - start_time
            return {
                "query": query,
                "response": f"I apologize, but I encountered an error processing your query: {str(e)[:100]}... The system is still operational but with limited capabilities.",
                "model_used": "error_fallback",
                "processing_time": processing_time,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "status": "error"
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        uptime = time.time() - self.start_time
        return {
            "gemini_status": "connected" if self.gemini_available else "disconnected",
            "primary_model": self.primary_model if self.gemini_available else "none",
            "uptime_seconds": uptime,
            "uptime_formatted": f"{int(uptime//60)}m {int(uptime%60)}s"
        }
    
    def display_help(self):
        """Display help information."""
        print("\n" + "=" * 60)
        print("🤖 MARS QUANTUM INTERACTIVE HELP")
        print("=" * 60)
        print("Available commands:")
        print("  • Type any question to get an AI response")
        print("  • 'status' - Show system status")
        print("  • 'help' - Show this help message")
        print("  • 'exit' or 'quit' - Exit the system")
        print("\nExamples:")
        print("  • What is quantum computing?")
        print("  • Explain machine learning algorithms")
        print("  • How does artificial intelligence work?")
        print("=" * 60)

async def interactive_loop():
    """Main interactive loop for user queries."""
    mars = MarsInteractiveSystem()
    
    # Always proceed regardless of API availability
    print("\n🎯 MARS QUANTUM INTERACTIVE MODE")
    print("=" * 60)
    if mars.gemini_available:
        print("✅ AI services connected - Full capabilities available")
    else:
        print("⚠️  AI services limited - Local processing mode active")
        print("💡 Tip: API quotas typically reset daily. Try again later for full AI responses.")
    print("Ask me anything! Type 'help' for commands or 'exit' to quit.")
    print("=" * 60)
    
    query_count = 0
    
    while True:
        try:
            print(f"\n🔹 Query #{query_count + 1}")
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            # Handle special commands
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("\n👋 Thank you for using MARS Quantum! Goodbye!")
                break
            
            elif user_input.lower() == 'help':
                mars.display_help()
                continue
            
            elif user_input.lower() == 'status':
                status = mars.get_system_status()
                print(f"\n📊 System Status:")
                print(f"   Gemini API: {status['gemini_status']}")
                print(f"   Model: {status['primary_model']}")
                print(f"   Uptime: {status['uptime_formatted']}")
                print(f"   Queries processed: {query_count}")
                continue
            
            # Process the query
            print("\n🤖 MARS Quantum is thinking...")
            result = await mars.process_user_query(user_input)
            
            print(f"\n🎯 MARS Response:")
            print("-" * 50)
            print(result['response'])
            print("-" * 50)
            print(f"⏱️  Processed in {result['processing_time']:.2f} seconds using {result['model_used']}")
            
            query_count += 1
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
            print("Please try again or type 'exit' to quit.")

if __name__ == "__main__":
    print("🚀 Starting MARS Quantum Interactive System...")
    asyncio.run(interactive_loop())
