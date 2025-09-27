# 🛒 Generative AI E-commerce Chatbot (Natural Language Product Discovery)

An intelligent shopping assistant that enables natural language product searches and recommendations grounded in your store's catalog. Built with a modular Flask-based web app, robust data handling, and AI-powered query understanding for seamless integration of e-commerce data with conversational AI.

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-Web_Framework-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![AI](https://img.shields.io/badge/Generative_AI-Natural_Language-green?style=for-the-badge&logo=openai&logoColor=white)](https://openai.com/)
[![E-commerce](https://img.shields.io/badge/E-commerce-Product_Search-orange?style=for-the-badge&logo=shopify&logoColor=white)](https://www.shopify.com/)

✨ **Features**

🗣️ **Conversational Product Q&A**: Ask questions in plain English about products, specs, categories, prices, and more—get tailored responses.

🔍 **Catalog-Grounded Responses**: Searches and retrieves from your product data (CSV/JSON in Data/) for accurate, relevant answers.

🎨 **Clean Web UI**: Simple, responsive interface with HTML templates and CSS/JS assets for an engaging chat experience.

🧩 **Modular Bot Logic**: Encapsulated in the `ecommercebot/` package for easy swapping of models, retrieval methods, or data loaders without UI changes.

🛡️ **Customizable & Safe**: Configure search behaviors, answer styles, and safety filters to prevent off-topic or unsafe responses.

🗂️ **Project structure**
```
.
├─ Data/                      # Product data (CSV/JSON/etc.) used to answer questions
├─ ecommercebot/              # Python package: retrieval, ranking, LLM helpers, utils
├─ static/                    # CSS/JS/assets for the web UI
├─ templates/                 # Jinja/HTML templates
├─ app.py                     # Web app entry point (run this)
├─ requirements.txt           # Python dependencies
└─ setup.py                   # Package metadata / editable install
```

🚀 **Quickstart**

**Prerequisites**
- Python 3.8 or higher
- Product catalog data (CSV/JSON format)
- Optional: API key for external LLM (e.g., OpenAI) for advanced query processing

**1) Environment setup**
```bash
git clone https://github.com/AbdullahRasheed45/GENERATIVE_AI-Ecommerce_Chatbot.git
cd GENERATIVE_AI-Ecommerce_Chatbot

# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .            # Optional: Install as editable package for development
```

**2) Configuration**
```bash
# Place your product catalog in Data/
# Example: Copy your products.csv to Data/products.csv

# Create environment file for API keys (optional)
cat > .env << EOF
# AI Provider Configuration
OPENAI_API_KEY=your_openai_key_here
MODEL_ID=gpt-4o-mini

# Application Settings
LOG_LEVEL=INFO
MAX_RESULTS=5
DEFAULT_CURRENCY=USD
EOF
```

**3) Run the application**
```bash
# Start the e-commerce chatbot
python app.py

# Access the web UI at http://127.0.0.1:5000
# Enter queries in the chat interface and get real-time responses
```

🧠 **System architecture**

**1. Natural Language Processing Layer**
```python
# Query understanding and intent extraction in ecommercebot/query_parser.py
def parse_product_query(user_input: str) -> ProductQuery:
    """Extract intent, entities, and constraints from natural language"""
    
    # Intent classification: search, compare, recommend
    intent = classify_intent(user_input)  # e.g., "search" or "compare"
    
    # Entity extraction: category, brand, price range
    entities = extract_entities(user_input)  # e.g., {"category": "headphones", "max_price": 100}
    
    # Constraint parsing: "under $60", "with good battery life"
    constraints = parse_constraints(user_input)
    
    return ProductQuery(intent=intent, entities=entities, constraints=constraints)
```

**2. Catalog Data Abstraction Layer**
```python
# ecommercebot/data_loader.py - Flexible catalog loader
class CatalogLoader:
    """Load and index product data from various formats"""
    
    def __init__(self, data_path: str = "Data/"):
        self.products = self._load_data(data_path)
        self.index = self._build_index(self.products)  # Vector store or keyword index
        
    def _load_data(self, path: str) -> List[Product]:
        """Support CSV, JSON, etc. with schema mapping"""
        if path.endswith(".csv"):
            df = pd.read_csv(path)
            return [Product.from_row(row) for _, row in df.iterrows()]
        # Add JSON/XML loaders as needed
        
    def retrieve_candidates(self, query: ProductQuery) -> List[Product]:
        """Hybrid retrieval: keyword + semantic search"""
        keywords = query.get_keywords()
        candidates = self.index.search(keywords, filters=query.constraints)
        return self._rank_candidates(candidates, query)
```

**3. Response Generation System**
```python
# ecommercebot/response_generator.py - AI-enhanced responses
class ResponseGenerator:
    """Generate natural, product-focused responses"""
    
    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm
        if use_llm:
            self.llm_client = OpenAIClient()
    
    def generate_response(self, products: List[Product], query: ProductQuery) -> str:
        """Create conversational summary from retrieved products"""
        
        if self.use_llm:
            return self._generate_llm_response(products, query)
        else:
            return self._generate_template_response(products, query)
    
    def _generate_llm_response(self, products: List[Product], query: ProductQuery) -> str:
        """Use LLM for friendly, grounded summaries"""
        prompt = f"""
        Based on these products:
        {self._format_products(products)}
        
        User asked: "{query.original_text}"
        
        Provide a helpful, conversational response listing matching items with key specs, prices, and reasons why they fit.
        Limit to {MAX_RESULTS} results. Keep it sales-friendly but factual.
        """
        return self.llm_client.generate(prompt)
```

**4. Error Handling and Logging**
```python
# ecommercebot/exceptions.py - Custom error management
class EcommerceBotException(Exception):
    """Base exception for chatbot errors"""
    
class CatalogLoadError(EcommerceBotException):
    """Raised when data loading fails"""
    
class QueryParseError(EcommerceBotException):
    """Raised when query cannot be understood"""
    
class NoResultsError(EcommerceBotException):
    """Raised when no products match"""

# ecommercebot/logger.py - Structured logging
def setup_logger() -> logging.Logger:
    """Configure logging for development and production"""
    logger = logging.getLogger("ecommerce_chatbot")
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    # File handler
    file_handler = logging.FileHandler("chatbot.log")
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    
    return logger
```

⚙️ **Configuration and customization**

**Catalog Schema Configuration**:
```python
# ecommercebot/config.py - Map your data columns
COLUMN_MAPPING = {
    'id': 'id',
    'name': 'title',
    'category': 'category',
    'description': 'description',
    'price': 'price',
    'currency': 'currency',
    'brand': 'brand',
    'image': 'image_url',
    'stock': 'stock'
}

# Search settings
RETRIEVAL_METHOD = 'hybrid'  # Options: 'keyword', 'semantic', 'hybrid'
MAX_RESULTS = 5
MIN_STOCK = 1  # Filter out-of-stock items
```

**Response Customization**:
```python
# Configurable response styles
RESPONSE_STYLES = {
    'concise': "Found {count} matches: {product_list}",
    'detailed': "Here are {count} options: {product_details} Reasons: {reasons}",
    'conversational': "Based on your query for {query_summary}, I'd recommend these: {product_list}. {additional_context}"
}

# Context-aware recommendations
def get_additional_context(products: List[Product]) -> str:
    """Add personalized tips"""
    contexts = []
    if any(p.price < 50 for p in products):
        contexts.append("Great budget options available!")
    if len(products) > 3:
        contexts.append("Plenty of choices—let me know if you need more filters.")
    return " ".join(contexts)
```

🗣️ **Natural language query examples**

**Basic Searches**:
- *"Wireless earbuds under $60 with good battery life."*
- *"Show running shoes from Brand X, men’s sizes 9–10."*
- *"Do you have a 4K monitor between 27–32 inches with USB-C?"*

**Comparisons and Recommendations**:
- *"Compare iPhone 14 cases that have MagSafe."*
- *"Best laptops for graphic design around $1000."*
- *"Affordable smartwatches with heart rate monitoring."*

**Filtered Queries**:
- *"Eco-friendly clothing brands in stock."*
- *"Gaming keyboards with RGB lighting over $50."*
- *"Summer dresses in blue or green, size medium."*

🔧 **Advanced features and extensions**

**Multi-Product Comparison**:
```python
# ecommercebot/comparator.py - Side-by-side comparisons
class ProductComparator:
    """Generate comparative tables for products"""
    
    def compare_products(self, products: List[Product], attributes: List[str]) -> str:
        """Create Markdown table for comparisons"""
        table = "| Product | " + " | ".join(attributes) + " |\n"
        table += "|---|" + "---|" * len(attributes) + "\n"
        
        for product in products:
            row = f"| {product.name} |"
            for attr in attributes:
                row += f" {getattr(product, attr, 'N/A')} |"
            table += row + "\n"
        
        return table
```

**Recommendation Engine**:
```python
# ecommercebot/recommender.py - Personalized suggestions
class Recommender:
    """Suggest related or upsell products"""
    
    def get_recommendations(self, product: Product, catalog: List[Product]) -> List[Product]:
        """Find similar items based on category and features"""
        similar = [p for p in catalog if p.category == product.category and p.id != product.id]
        return sorted(similar, key=lambda p: abs(p.price - product.price))[:3]  # Sort by price similarity
```

**Multi-language Support**:
```python
# ecommercebot/i18n.py - Internationalization
class MultiLanguageBot:
    """Handle queries in multiple languages"""
    
    def __init__(self, default_language: str = "en"):
        self.translator = TranslationClient()
        
    async def process_query(self, query: str, language: str = None) -> str:
        detected = language or self.detect_language(query)
        if detected != "en":
            english_query = await self.translator.translate(query, "en")
        else:
            english_query = query
            
        response = await self.bot.process(english_query)
        
        if detected != "en":
            return await self.translator.translate(response, detected)
        return response
```

🧪 **Development and testing framework**

**Interactive Testing**:
```python
# Example in ecommercebot/tests/test_query.py
def test_query_parsing():
    """Validate query understanding"""
    test_queries = [
        "Wireless earbuds under $60",
        "Compare MagSafe cases"
    ]
    
    for query in test_queries:
        parsed = parse_product_query(query)
        print(f"Query: {query}")
        print(f"Intent: {parsed.intent}")
        print(f"Entities: {parsed.entities}")
        print("---")

def evaluate_response_quality():
    """Assess response relevance"""
    sample_products = [Product(name="Earbuds", price=49.99)]
    query = ProductQuery(intent="search", entities={"category": "audio"})
    response = generate_response(sample_products, query)
    print(response)
```

**Performance Monitoring**:
```python
# ecommercebot/monitor.py - Track usage
class BotMonitor:
    """Monitor query performance and patterns"""
    
    def log_query(self, query: str, response: str, time_taken: float):
        self.metrics.increment('queries_total')
        self.metrics.histogram('response_time', time_taken)
        if "no results" in response.lower():
            self.metrics.increment('no_results')
```

🔒 **Production considerations**

**Security and Input Validation**:
```python
# ecommercebot/security.py - Protect against injections
class SecurityManager:
    """Validate inputs and prevent unsafe operations"""
    
    def validate_query(self, query: str) -> bool:
        """Check for malicious patterns"""
        if any(keyword in query.lower() for keyword in ['drop table', 'delete', 'script']):
            raise SecurityError("Invalid query detected")
        return True
```

**Caching and Performance**:
```python
# ecommercebot/cache.py - Optimize retrieval
class ProductCache:
    """In-memory caching for frequent queries"""
    
    def __init__(self):
        self.cache = {}
        
    def get_cached_results(self, query_hash: str) -> Optional[List[Product]]:
        if query_hash in self.cache:
            return self.cache[query_hash]
        return None
    
    def set_cache(self, query_hash: str, results: List[Product]):
        self.cache[query_hash] = results
```

🐛 **Troubleshooting guide**

**Common Configuration Issues**:
- **App Doesn’t Start** → Ensure `pip install -r requirements.txt` completed; try a clean venv.
- **No Products Returned** → Confirm CSV/JSON in Data/ matches column mapping.
- **Static Files Not Loading** → Use `url_for('static', filename='...')` in templates.
- **LLM Errors** → Verify API keys in .env and model availability.

**Performance and Response Issues**:
- **Slow Queries** → Enable caching or optimize index building.
- **Irrelevant Results** → Refine retrieval method or add more filters.
- **Memory Usage** → Limit max results and clear cache periodically.

**Development and Debugging**:
- **Import Errors** → Activate venv and reinstall dependencies.
- **Template Issues** → Check Jinja syntax in templates/.
- **Logging Problems** → Ensure log file permissions.

📚 **Learning resources and roadmap**

**Technical Concepts Covered**:
- **API/Web App Patterns**: Flask routing and templating.
- **Natural Language Processing**: Intent/entity extraction.
- **Data Handling**: CSV/JSON parsing and indexing.
- **AI Integration**: LLM for response generation.

**Future Enhancement Ideas**:
- **Faceted Filters**: UI sliders for price, brand toggles.
- **Image Thumbnails**: Display product images in responses.
- **Cart Integration**: Emit IDs for front-end cart addition.
- **Re-ranking**: Use sales data for better results.
- **Multilingual**: Auto-detect and translate queries.

📜 **License**

MIT License - see [LICENSE](LICENSE) file for complete terms.

## 📞 Connect & Support

<div align="center">

### 🚀 Ready to Build Intelligent E-commerce Experiences?

[![Portfolio](https://img.shields.io/badge/Portfolio-000000?style=for-the-badge&logo=About.me&logoColor=white)](https://techvibes360.com)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/abdullahrasheed-/)
[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:abdullahrasheed45@gmail.com)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/AbdullahRasheed45)

**Let's make shopping more conversational and smart!**

</div>

---

*Built with ❤️ for developers interested in AI-driven e-commerce. Perfect for learning natural language search, data indexing, and web app development.*
