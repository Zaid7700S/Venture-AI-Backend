import os
import logging
import asyncio
import copy
import enum
from typing import List, Dict, Any, TypedDict, Optional, Annotated
from datetime import datetime
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from ddgs import DDGS
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

# FastAPI imports
from fastapi import FastAPI, HTTPException, Header, Request, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import uvicorn

# ==========================================
# LOGGING & CONFIGURATION
# ==========================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)

class Settings(BaseSettings):
    contact_email: str = "admin@example.com"
    osm_radius_m: int = 5000
    search_max_results: int = 3
    default_sq_ft: int = 400
    utility_as_rent_pct: float = 0.3
    api_key: str = "secret-key-change-me" # Default API key for auth

    class Config:
        env_file = ".env"

load_dotenv()
settings = Settings()

# ==========================================
# UTILITIES & SCRAPERS (ASYNC)
# ==========================================

async def duckduckgo_search(query: str, max_results: int = None) -> str:
    max_res = max_results or settings.search_max_results
    def _sync_search():
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_res))
                if not results:
                    return "No results found."
                return "\n".join([f"- {r['title']}: {r['body']}" for r in results])
        except Exception as e:
            logger.error(f"DDG Search failed for query '{query}': {e}")
            return "Search unavailable."
    return await asyncio.to_thread(_sync_search)

async def geocode_location(query: str) -> Dict[str, Any]:
    if "remote" in query.lower() or "online" in query.lower():
        return {"lat": 0.0, "lon": 0.0, "display_name": "Remote"}
    
    headers = {'User-Agent': f'AIVentureBuilder/6.0 (contact: {settings.contact_email})'}
    url = f"https://nominatim.openstreetmap.org/search?q={httpx.URL(query)}&format=json"
    try:
        async with httpx.AsyncClient(headers=headers, timeout=10.0) as client:
            res = await client.get(url)
            res.raise_for_status()
            data = res.json()
            if data:
                return {"lat": float(data[0]["lat"]), "lon": float(data[0]["lon"]), "display_name": data[0]["display_name"]}
    except Exception as e:
        logger.error(f"Geocoding Error for '{query}': {e}")
    return {"lat": 0.0, "lon": 0.0, "display_name": query}

async def fetch_osm_competitors(lat: float, lon: float, biz_keyword: str, radius: int = None) -> List[Dict[str, Any]]:
    if lat == 0.0 and lon == 0.0:
        return []
    
    rad = radius or settings.osm_radius_m
    overpass_url = "https://overpass-api.de/api/interpreter"
    query = f'[out:json];(node(around:{rad},{lat},{lon})["shop"~"{biz_keyword}",i];node(around:{rad},{lat},{lon})["amenity"~"{biz_keyword}",i];node(around:{rad},{lat},{lon})["leisure"~"{biz_keyword}",i];);out body 10;'
    headers = {'User-Agent': f'AIVentureBuilder/6.0 (business_planner)'} 
    
    try:
        async with httpx.AsyncClient(headers=headers, timeout=10.0) as client:
            res = await client.get(overpass_url, params={'data': query})
            res.raise_for_status()
            data = res.json()
            
            competitors = []
            for elem in data.get('elements', []):
                tags = elem.get('tags', {})
                if "name" in tags:
                    biz_type = tags.get("shop") or tags.get("amenity") or tags.get("leisure") or "business"
                    competitors.append({"name": tags.get("name"), "type": biz_type})
            return competitors
    except Exception as e:
        logger.error(f"OSM API Error: {e}")
        return []

# ==========================================
# PYDANTIC STRUCTURED SCHEMAS & ENUMS
# ==========================================

class BusinessType(str, enum.Enum):
    RETAIL = "Retail/Product"
    SERVICE = "Service/Experience"

class SupervisorSchema(BaseModel):
    neighborhood: str = Field(description="Specific neighborhood or area")
    city_country: str = Field(description="City and Country")
    currency: str = Field(description="Local currency code e.g. PKR, USD")
    is_remote: bool = Field(default=False, description="true if online/digital, false if physical")

class BlueprintTierSpec(BaseModel):
    tier_name: str
    estimated_sq_ft: int = Field(default=400)
    key_equipment_categories: List[str]
    core_roles: List[str]
    estimated_startup_capex: float = Field(default=0.0)

class MasterBlueprintSchema(BaseModel):
    business_summary: str
    business_type: BusinessType
    primary_commodity_material: str = Field(description="Primary raw material e.g. 22K Gold. If Service, write 'N/A'")
    bootstrapped_tier: BlueprintTierSpec
    standard_tier: BlueprintTierSpec
    premium_tier: BlueprintTierSpec

class MarketPricingSchema(BaseModel):
    primary_material: str
    unit_price: float = Field(default=0.0)
    unit_measurement: str = Field(default="unit")

class CompetitorItem(BaseModel):
    name: str
    offering: str
    gap: str

class CompetitorSchema(BaseModel):
    competitors: List[CompetitorItem]

class LocationSchema(BaseModel):
    estimated_monthly_rent: float = Field(default=0.0)
    notes: str

class AssetItem(BaseModel):
    item: str
    cost: float = Field(default=0.0)

class AssetSchema(BaseModel):
    total_asset_cost: float = Field(default=0.0)
    breakdown: List[AssetItem]

class PermitItem(BaseModel):
    permit_name: str
    authority: str
    estimated_cost: float = Field(default=0.0)

class LegalSchema(BaseModel):
    permits: List[PermitItem]

class RoleItem(BaseModel):
    title: str
    headcount: int = Field(default=1)
    monthly_salary_per_person: float = Field(default=0.0)

class WorkforceSchema(BaseModel):
    roles: List[RoleItem]
    total_monthly_payroll: float = Field(default=0.0)

class ProductItem(BaseModel):
    item: str
    retail_price: float = Field(default=0.0)
    cogs: float = Field(default=0.0)
    description: str

class ProductSchema(BaseModel):
    products: List[ProductItem]

class CampaignItem(BaseModel):
    campaign_name: str
    execution_steps: str
    expected_roi: str

class MarketingSchema(BaseModel):
    brand_names: List[str]
    tailored_campaigns: List[CampaignItem]

# ==========================================
# STATE SCHEMA
# ==========================================

class VentureState(TypedDict):
    business_idea: str
    raw_location_input: str
    neighborhood: str
    city_country: str
    budget: float
    currency: str
    is_remote: bool
    business_type: str 
    blueprint: Dict[str, Any]
    selected_tier: Dict[str, Any]
    competitors_data: List[Dict[str, Any]]
    location_data: Dict[str, Any]
    market_rates: Dict[str, Any]
    asset_data: Dict[str, Any]
    legal_requirements: List[Dict[str, Any]]
    workforce_data: Dict[str, Any]
    product_menu: List[Dict[str, Any]]
    financial_model: Dict[str, Any]
    marketing_plan: Dict[str, Any]
    is_feasible: bool
    pivot_suggestion: str
    final_plan_markdown: str
    search_failed: bool

# ==========================================
# SAFE LLM INVOCATION WRAPPER
# ==========================================

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=2, max=10),
    reraise=True,
)
async def safe_structured_invoke(schema, prompt, llm_instance):
    """Wrapper for LLM structured invocation with Tenacity retries."""
    try:
        return await llm_instance.with_structured_output(schema).ainvoke([HumanMessage(content=prompt)])
    except Exception as e:
        logger.error(f"LLM Structured Output Failed: {e}. Retrying...")
        raise

def safe_float(val: Any) -> float:
    try:
        return float(val) if val is not None else 0.0
    except (ValueError, TypeError):
        return 0.0

# ==========================================
# AGENT NODES (ASYNC & PARALLELIZED)
# ==========================================

async def supervisor_node(state: VentureState, config: RunnableConfig) -> Dict[str, Any]:
    user_llm = config["configurable"]["llm"]
    prompt = f"Deconstruct location string '{state['raw_location_input']}' for business '{state['business_idea']}'. Separate neighborhood from city/country."
    res = await safe_structured_invoke(SupervisorSchema, prompt, user_llm)
    data = res.model_dump()
    return {
        "neighborhood": data["neighborhood"],
        "city_country": data["city_country"],
        "currency": data["currency"],
        "is_remote": data["is_remote"]
    }

async def blueprint_agent_node(state: VentureState, config: RunnableConfig) -> Dict[str, Any]:
    user_llm = config["configurable"]["llm"]
    prompt = f"""You are the Master Venture Architect. Create a master operational blueprint for a '{state['business_idea']}' in {state['neighborhood']}, {state['city_country']}.
    Classify the business_type as either 'Retail/Product' or 'Service/Experience'.
    Output 3 distinct budget tiers (Bootstrapped, Standard, Premium) with categorical equipment/role specifications.
    CRITICAL: All numeric fields must be valid numbers (ints or floats). Default capex to 0.0 if unknown."""
    
    res = await safe_structured_invoke(MasterBlueprintSchema, prompt, user_llm)
    dumped = res.model_dump()
    return {"blueprint": dumped, "business_type": dumped["business_type"].value}

async def financial_analyst_node(state: VentureState, config: RunnableConfig) -> Dict[str, Any]:
    blueprint = state["blueprint"]
    budget = state["budget"]
    
    tiers = sorted([
        blueprint["bootstrapped_tier"],
        blueprint["standard_tier"],
        blueprint["premium_tier"]
    ], key=lambda t: safe_float(t.get("estimated_startup_capex", 0.0)))
    
    if any(safe_float(t.get("estimated_startup_capex", 0.0)) <= 0.0 for t in tiers):
        logger.warning("LLM returned 0 or invalid capex for a tier.")
        
    boot, std, prem = tiers[0], tiers[1], tiers[2]
    
    if budget >= prem["estimated_startup_capex"]:
        selected = prem
    elif budget >= std["estimated_startup_capex"]:
        selected = std
    else:
        selected = boot
        
    is_feasible = budget >= selected["estimated_startup_capex"]
    pivot = "" if is_feasible else f"Budget ({budget:,.0f} {state['currency']}) is below the minimum Bootstrapped setup ({boot['estimated_startup_capex']:,.0f} {state['currency']})."
    
    selected_copy = copy.deepcopy(selected)
    
    return {
        "selected_tier": selected_copy,
        "financial_model": {"capex": selected_copy["estimated_startup_capex"], "status": "PASS" if is_feasible else "FAIL"},
        "is_feasible": is_feasible,
        "pivot_suggestion": pivot
    }

async def market_pricing_node(state: VentureState, config: RunnableConfig) -> Dict[str, Any]:
    user_llm = config["configurable"]["llm"]
    if state.get("business_type") == BusinessType.SERVICE.value:
        return {"market_rates": {"primary_material": "Service-Based (No Raw Materials)", "unit_price": 0.0, "unit_measurement": "N/A"}}

    material = state["blueprint"].get("primary_commodity_material", "Raw Material")
    current_year = datetime.now().year
    query = f"current wholesale rate of {material} in {state['city_country']} {state['currency']} {current_year}"
    search_data = await duckduckgo_search(query)
    
    prompt = f"""Extract the current rate for {material} in {state['currency']} from the web search data.
    CRITICAL: 'unit_price' MUST be a numeric float (e.g. 245000.0). Do NOT output explanatory text.
    Web Search Data: {search_data}"""
    
    res = await safe_structured_invoke(MarketPricingSchema, prompt, user_llm)
    return {"market_rates": res.model_dump()}

async def competitor_analyst_node(state: VentureState, config: RunnableConfig) -> Dict[str, Any]:
    user_llm = config["configurable"]["llm"]
    if state["is_remote"]:
        return {"competitors_data": []}
        
    biz_keyword = state['business_idea'].split()[0].lower()
    coords = await geocode_location(f"{state['neighborhood']}, {state['city_country']}")
    osm = await fetch_osm_competitors(coords["lat"], coords["lon"], biz_keyword)
    
    query = f"{state['business_idea']} in {state['neighborhood']}, {state['city_country']} reviews competitors"
    search_data = await duckduckgo_search(query)
    
    prompt = f"Identify 3 competitors for {state['business_idea']} in {state['neighborhood']}, {state['city_country']}. Detail their market gaps. OSM: {osm}. Web: {search_data}"
    res = await safe_structured_invoke(CompetitorSchema, prompt, user_llm)
    return {"competitors_data": res.model_dump()["competitors"]}

async def location_analyst_node(state: VentureState, config: RunnableConfig) -> Dict[str, Any]:
    user_llm = config["configurable"]["llm"]
    if state["is_remote"]:
        return {"location_data": {"estimated_monthly_rent": 0.0, "notes": "Remote Digital Business."}}
        
    query = f"commercial shop rent per sq ft in {state['neighborhood']}, {state['city_country']}"
    rent_data = await duckduckgo_search(query)
    
    sq_ft = state["selected_tier"].get("estimated_sq_ft", settings.default_sq_ft)
    prompt = f"""Estimate monthly rent for a {sq_ft} sq ft space in {state['neighborhood']}, {state['city_country']} in {state['currency']}.
    CRITICAL: 'estimated_monthly_rent' MUST be a number. Data: {rent_data}"""
    
    res = await safe_structured_invoke(LocationSchema, prompt, user_llm)
    return {"location_data": res.model_dump()}

async def legal_agent_node(state: VentureState, config: RunnableConfig) -> Dict[str, Any]:
    user_llm = config["configurable"]["llm"]
    query = f"business registration permits required for {state['business_idea']} in {state['city_country']}"
    search_data = await duckduckgo_search(query)
    
    prompt = f"""Checklist of required local/national permits for {state['business_idea']} in {state['city_country']}.
    Ensure these are actual permits in {state['city_country']}. All cost fields MUST be valid numbers. Data: {search_data}"""
    
    res = await safe_structured_invoke(LegalSchema, prompt, user_llm)
    return {"legal_requirements": res.model_dump()["permits"]}

async def workforce_analyst_node(state: VentureState, config: RunnableConfig) -> Dict[str, Any]:
    user_llm = config["configurable"]["llm"]
    roles = ", ".join(state["selected_tier"].get("core_roles", []))
    query = f"average salary for {roles} in {state['city_country']} {state['currency']}"
    search_data = await duckduckgo_search(query)
    
    prompt = f"""Calculate realistic monthly payroll for the team ({roles}) running a {state['business_idea']} in {state['neighborhood']}, {state['city_country']}.
    
    CRITICAL SALARY GUARDRAILS:
    Base the salaries on the local economic standards of {state['city_country']} and the web search data provided. 
    Do NOT output corporate enterprise-level salaries. Headcount and salary MUST be realistic valid numbers in {state['currency']}. 
    Data: {search_data}"""
    
    res = await safe_structured_invoke(WorkforceSchema, prompt, user_llm)
    work_data = res.model_dump()
    
    calculated_payroll = sum(safe_float(role.get('headcount', 0)) * safe_float(role.get('monthly_salary_per_person', 0)) for role in work_data.get('roles', []))
    work_data['total_monthly_payroll'] = calculated_payroll
    
    return {"workforce_data": work_data}

async def asset_equipment_node(state: VentureState, config: RunnableConfig) -> Dict[str, Any]:
    user_llm = config["configurable"]["llm"]
    tier = state["selected_tier"]
    categories = ", ".join(tier.get("key_equipment_categories", []))
    
    query = f"commercial setup price for {categories} in {state['city_country']} {state['currency']}"
    search_data = await duckduckgo_search(query)
    
    prompt = f"""Estimate pricing for the selected tier '{tier['tier_name']}' equipment ({categories}) for {state['business_idea']}.
    CRITICAL CURRENCY WARNING: The required currency is {state['currency']}. 
    If web prices are in USD, convert them to {state['currency']} using current exchange rates. 
    If the business is Retail/Product, include Initial Inventory stock costs. If Service/Experience, only include fixed asset/equipment costs.
    Data: {search_data}
    """
    
    res = await safe_structured_invoke(AssetSchema, prompt, user_llm)
    asset_data = res.model_dump()
    
    calculated_total = sum(safe_float(item.get('cost', 0.0)) for item in asset_data.get('breakdown', []))
    asset_data['total_asset_cost'] = calculated_total
    
    return {"asset_data": asset_data}

async def product_strategist_node(state: VentureState, config: RunnableConfig) -> Dict[str, Any]:
    user_llm = config["configurable"]["llm"]
    if state.get("business_type") == BusinessType.SERVICE.value:
        prompt_instruction = """The business is a Service/Experience. Create 4 Service Offerings (e.g. Hourly Passes, Monthly Memberships, VIP Sessions). 
        DO NOT try to sell the physical core equipment. COGS for services is usually low (electricity, maintenance)."""
    else:
        prompt_instruction = f"""The business is Retail/Product. Develop 4 premium products to sell. 
        Base COGS strictly on raw material rates: {state.get('market_rates', {})}."""

    prompt = f"""Develop 4 revenue streams/products for {state['business_idea']} in {state['currency']}.
    {prompt_instruction}
    CRITICAL: All prices and COGS fields MUST be valid numeric floats."""
    
    res = await safe_structured_invoke(ProductSchema, prompt, user_llm)
    return {"product_menu": res.model_dump()["products"]}

async def marketing_agent_node(state: VentureState, config: RunnableConfig) -> Dict[str, Any]:
    user_llm = config["configurable"]["llm"]
    prompt = f"""Growth strategy for '{state['business_idea']}' in {state['neighborhood']}, {state['city_country']}.
    1. Generate 3 highly creative, unique brand names.
    2. Target the specific demographic of {state['neighborhood']}. Provide 3 creative, real-world launch campaigns.
    Rule: NO high-tech AR/AI gimmicks if this is a physical store. Use real-world creative tactics."""
    
    res = await safe_structured_invoke(MarketingSchema, prompt, user_llm)
    return {"marketing_plan": res.model_dump()}

async def markdown_compiler_node(state: VentureState, config: RunnableConfig) -> Dict[str, Any]:
    tier = state.get("selected_tier", {})
    rent = safe_float(state.get("location_data", {}).get("estimated_monthly_rent"))
    payroll = safe_float(state.get("workforce_data", {}).get("total_monthly_payroll"))
    opex = rent + payroll + (rent * settings.utility_as_rent_pct)
    
    primary_material = state.get('market_rates', {}).get('primary_material', 'N/A')
    unit_price = safe_float(state.get('market_rates', {}).get('unit_price'))
    unit_measurement = state.get('market_rates', {}).get('unit_measurement', 'unit')
    
    total_asset_cost = safe_float(state.get('asset_data', {}).get('total_asset_cost'))
    estimated_startup_capex = safe_float(tier.get("estimated_startup_capex"))

    plan = f"""# 🚀 Detailed Business Plan & Feasibility Study

## 📌 Executive Summary
* **Business Concept:** {state['business_idea']}
* **Business Category:** {state.get('business_type', 'N/A')}
* **Selected Model Tier:** **{tier.get('tier_name', 'Standard')} Tier**
* **Target Neighborhood:** {state['neighborhood']}
* **City & Region:** {state['city_country']}
* **Available User Budget:** {state['budget']:,.0f} {state['currency']}
* **Feasibility Status:** {'✅ **FEASIBLE**' if state['is_feasible'] else '⚠️ **INSUFFICIENT BUDGET**'}

---

## 🏬 Real Estate & Neighborhood Dynamics ({state['neighborhood']})
* **Estimated Monthly Rent:** {rent:,.0f} {state['currency']}
* **Store Footprint Size:** ~{tier.get('estimated_sq_ft', settings.default_sq_ft)} sq ft
* **Neighborhood Assessment:** {state.get('location_data', {}).get('notes', 'N/A')}

---

## 📈 Commodity & Raw Material Benchmark
* **Primary Raw Commodity:** {primary_material}
* **Current Rate:** {unit_price:,.0f} {state['currency']} per {unit_measurement}

---

## 🛠️ Capital Expenditure (CAPEX) & Asset Fit-Out
* **Total Estimated Equipment & Stock Setup:** **{total_asset_cost:,.0f} {state['currency']}**

**Asset Breakdown ({tier.get('tier_name', 'Standard')} Tier):**
"""
    for asset in state.get('asset_data', {}).get('breakdown', []):
        cost = safe_float(asset.get('cost'))
        plan += f"- **{asset.get('item')}**: ~{cost:,.0f} {state['currency']}\n"

    plan += f"\n--- \n## 🥊 Competitor Analysis & Market Gaps ({state['neighborhood']})\n"
    for comp in state.get('competitors_data', []):
        plan += f"### 🛡️ {comp.get('name')}\n* **Offering:** {comp.get('offering')}\n* **Market Gap:** *{comp.get('gap')}*\n\n"

    plan += f"\n--- \n## 👥 Team & Shift Workforce Operations\n"
    plan += f"* **Total Monthly Payroll:** {payroll:,.0f} {state['currency']}\n\n"
    for role in state.get('workforce_data', {}).get('roles', []):
        salary = safe_float(role.get('monthly_salary_per_person'))
        plan += f"- **{role.get('title')}** (x{role.get('headcount')}): {salary:,.0f} {state['currency']}/month\n"

    plan += f"\n--- \n## 💰 Master Financial Overview\n"
    plan += f"* **Total Startup CAPEX:** {estimated_startup_capex:,.0f} {state['currency']}\n"
    plan += f"* **Estimated Monthly OPEX:** {opex:,.0f} {state['currency']}/month *(Rent + Payroll + Utilities)*\n\n"
    
    if not state['is_feasible']:
        plan += f"> ⚠️ **Budget Warning:** {state.get('pivot_suggestion')}\n\n"

    plan += f"\n--- \n## 💎 Signature Offerings & Pricing\n"
    for item in state.get('product_menu', []):
        retail = safe_float(item.get('retail_price'))
        cogs = safe_float(item.get('cogs'))
        plan += f"### {item.get('item')}\n"
        plan += f"* *{item.get('description')}*\n"
        plan += f"* **Retail Price:** {retail:,.0f} {state['currency']} | **COGS / Delivery Cost:** {cogs:,.0f} {state['currency']}\n\n"

    plan += f"\n--- \n## 📜 Regulatory & Legal Checklist\n"
    for leg in state.get('legal_requirements', []):
        est_cost = safe_float(leg.get('estimated_cost'))
        plan += f"- [ ] **{leg.get('permit_name')}** ({leg.get('authority')}): Approx {est_cost:,.0f} {state['currency']}\n"

    plan += f"\n--- \n## 🚀 Targeted Launch Strategy ({state['neighborhood']})\n"
    brand_names = state.get('marketing_plan', {}).get('brand_names', [])
    plan += f"**Brand Identities:** {', '.join(brand_names) if brand_names else 'N/A'}\n\n"
    
    for camp in state.get('marketing_plan', {}).get('tailored_campaigns', []):
        plan += f"#### 🎯 {camp.get('campaign_name')}\n"
        plan += f"**Execution Strategy:** {camp.get('execution_steps')}\n"
        plan += f"**Expected Impact:** *{camp.get('expected_roi')}*\n\n"

    return {"final_plan_markdown": plan}

# ==========================================
# GRAPH ROUTING (PARALLELIZED)
# ==========================================

builder = StateGraph(VentureState)

builder.add_node("supervisor", supervisor_node)
builder.add_node("blueprint_agent", blueprint_agent_node)
builder.add_node("financial_analyst", financial_analyst_node)
builder.add_node("market_pricing", market_pricing_node)
builder.add_node("competitor_analyst", competitor_analyst_node)
builder.add_node("location_analyst", location_analyst_node)
builder.add_node("asset_equipment", asset_equipment_node)
builder.add_node("legal_agent", legal_agent_node)
builder.add_node("workforce_analyst", workforce_analyst_node)
builder.add_node("product_strategist", product_strategist_node)
builder.add_node("marketing_agent", marketing_agent_node)
builder.add_node("compiler", markdown_compiler_node)

builder.add_edge(START, "supervisor")
builder.add_edge("supervisor", "blueprint_agent")
builder.add_edge("blueprint_agent", "financial_analyst")

builder.add_edge("financial_analyst", "market_pricing")
builder.add_edge("financial_analyst", "competitor_analyst")
builder.add_edge("financial_analyst", "location_analyst")
builder.add_edge("financial_analyst", "legal_agent")
builder.add_edge("financial_analyst", "workforce_analyst")

builder.add_edge("market_pricing", "asset_equipment")
builder.add_edge("competitor_analyst", "asset_equipment")
builder.add_edge("location_analyst", "asset_equipment")
builder.add_edge("legal_agent", "asset_equipment")
builder.add_edge("workforce_analyst", "asset_equipment")

builder.add_edge("asset_equipment", "product_strategist")
builder.add_edge("product_strategist", "marketing_agent")
builder.add_edge("marketing_agent", "compiler")
builder.add_edge("compiler", END)

venture_builder_app = builder.compile()

# ==========================================
# FASTAPI SERVER & FRONTEND HOSTING
# ==========================================
limiter = Limiter(key_func=get_remote_address)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global INDEX_HTML
    index_path = Path("index.html")
    INDEX_HTML = index_path.read_text(encoding="utf-8") if index_path.exists() else "<h1>AI Venture Builder API</h1>"
    yield

app = FastAPI(lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class PlanRequest(BaseModel):
    business_idea: str = Field(..., min_length=3, max_length=200)
    raw_location_input: str = Field(..., min_length=2)
    budget: float = Field(..., gt=0)

@app.get("/")
async def get_homepage():
    return HTMLResponse(content=INDEX_HTML)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/generate")
@limiter.limit("5/minute")
async def generate_plan(data: PlanRequest, request: Request):
    # 1. Get the user's Groq API key from the frontend headers
    user_groq_key = request.headers.get("X-Groq-API-Key")
    if not user_groq_key:
        raise HTTPException(status_code=401, detail="Groq API Key is missing. Please add it in the UI.")
    
    # 2. Dynamically initialize the LLM for THIS user
    user_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3, groq_api_key=user_groq_key)
    
    try:
        initial_input = {
            "business_idea": data.business_idea,
            "raw_location_input": data.raw_location_input,
            "budget": data.budget,
            "neighborhood": "", "city_country": "", "currency": "USD", "is_remote": False,
            "business_type": "", "blueprint": {}, "selected_tier": {}, "competitors_data": [], 
            "location_data": {}, "market_rates": {}, "asset_data": {}, "legal_requirements": [], 
            "workforce_data": {}, "product_menu": [], "financial_model": {}, "marketing_plan": {}, 
            "is_feasible": False, "pivot_suggestion": "", "final_plan_markdown": "", "search_failed": False
        }

        # 3. Pass the user's LLM into the LangGraph config
        final_state = await venture_builder_app.ainvoke(
            initial_input, 
            {"configurable": {"llm": user_llm}}
        )
        
        safe_folder_name = "".join(c for c in final_state['business_idea'] if c.isascii() and (c.isalnum() or c in (' ', '_'))).strip() or "Business"
        base_dir = os.path.abspath("generated_plans")
        folder_path = os.path.abspath(os.path.join(base_dir, safe_folder_name))
        
        if not folder_path.startswith(base_dir):
            raise ValueError("Invalid path")
            
        os.makedirs(folder_path, exist_ok=True)
        
        version = 1
        while True:
            file_name = f"BusinessPlan_v{version}.md"
            file_path = os.path.join(folder_path, file_name)
            try:
                with open(file_path, "x", encoding="utf-8") as file:
                    file.write(final_state["final_plan_markdown"])
                break
            except FileExistsError:
                version += 1
        
        return JSONResponse(content={
            "markdown": final_state["final_plan_markdown"],
            "saved_file_path": file_path
        })
    
    except Exception as e:
        logger.exception(f"Server Error generating plan: {e}")
        return JSONResponse(content={"error": "An internal server error occurred while generating the plan."}, status_code=500)

# ==========================================
# CHAT ENDPOINT
# ==========================================

class ChatRequest(BaseModel):
    plan_markdown: str
    user_message: str
    chat_history: List[Dict[str, str]] = []

@app.post("/chat")
async def chat_with_plan(chat_data: ChatRequest, request: Request):
    user_groq_key = request.headers.get("X-Groq-API-Key")
    if not user_groq_key:
        raise HTTPException(status_code=401, detail="Groq API Key is missing.")
    
    user_llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.3, groq_api_key=user_groq_key)
    
    try:
        system_prompt = f"""You are an expert business analyst. The user has generated the following business plan:
        
        {chat_data.plan_markdown}
        
        Answer the user's follow-up questions concisely based on this plan. Use Markdown formatting (lists, bold) if appropriate."""
        
        messages = [SystemMessage(content=system_prompt)]
        
        for msg in chat_data.chat_history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))
                
        messages.append(HumanMessage(content=chat_data.user_message))
        
        response = await user_llm.ainvoke(messages)
        return {"response": response.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
