from src.core.config import get_settings

settings = get_settings()

print(f"LLM Provider : {settings.llm.provider}")
print(f"LLM Model    : {settings.llm.anthropic_model}")
print(f"CT page size : {settings.ct.page_size}")
print(f"Vector store : {settings.vs.store_type}")
print(f"API port     : {settings.api.port}")
print(f"Log level    : {settings.logging.level}")
print("\n✅ Config loaded successfully")