# AdaptXAI - System Design Document

**Version 1.0** | **AWS AI for Bharat Hackathon** | **February 2026**

---

## 1. Architecture Overview

### 1.1 System Layers

```
┌─────────────────────────────────────────────────────────┐
│  LAYER 6: Output (React Dashboard, Streamlit)          │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│  LAYER 5: What-If Interactive (Real-time Simulation)   │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│  LAYER 4: Feedback (User Ratings & Analytics)          │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│  LAYER 3: Personalization (Role-Based Adaptation)      │
│           Developer (Complexity=10) | EndUser (=2)      │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│  LAYER 2: Explanation Generator (SHAP/LIME)            │
└────────────────────┬────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│  LAYER 1: Input (Model & Dataset Ingestion)            │
└─────────────────────────────────────────────────────────┘
                     │
┌────────────────────▼────────────────────────────────────┐
│  DATA: PostgreSQL | Redis Cache | S3 Model Store       │
└─────────────────────────────────────────────────────────┘
```

---

## 2. Data Flow

### 2.1 Explanation Generation Flow

```mermaid
sequenceDiagram
    User->>API: POST /explain/local
    API->>Auth: Validate JWT
    Auth-->>API: User Role
    API->>Cache: Check cache
    alt Cache Miss
        API->>Explainer: Generate SHAP/LIME
        Explainer-->>API: Raw explanation
        API->>Cache: Store result
    end
    API->>Personalizer: Adapt for role
    Personalizer-->>API: Personalized view
    API-->>User: Return explanation
```

### 2.2 What-If Simulation Flow

```mermaid
sequenceDiagram
    User->>WebSocket: Connect
    loop Real-time
        User->>WebSocket: Modify features
        WebSocket->>Model: Predict (original)
        WebSocket->>Model: Predict (modified)
        WebSocket->>Explainer: Explain modified
        WebSocket-->>User: Comparison result
    end
```

---

## 3. API Design

### 3.1 Core Endpoints

```python
# backend/main.py
from fastapi import FastAPI, Depends, WebSocket

app = FastAPI(title="AdaptXAI API", version="1.0.0")

# Explanation endpoints
@app.post("/api/v1/explain/local")
async def explain_local(
    request: ExplainRequest,
    user = Depends(get_current_user)
):
    """Generate local explanation with role-based personalization"""
    # Load model
    model, features = await load_model(request.model_id)
    
    # Generate explanation
    explainer = ExplanationEngine(model, X_train, features)
    explanation = explainer.generate_local_explanation(
        np.array(request.instance), 
        method=request.method
    )
    
    # Personalize based on role
    personalizer = PersonalizationService()
    result = personalizer.adapt_explanation(explanation, user["role"])
    
    return result

@app.post("/api/v1/explain/global")
async def explain_global(request: GlobalExplainRequest, user = Depends(get_current_user)):
    """Generate global feature importance"""
    model, features = await load_model(request.model_id)
    explainer = ExplanationEngine(model, X_train, features)
    return explainer.generate_global_explanation()

# What-if simulation
@app.post("/api/v1/whatif/simulate")
async def whatif_simulate(simulation: WhatIfSimulation, user = Depends(get_current_user)):
    """Run what-if simulation"""
    model, features = await load_model(simulation.model_id)
    whatif_service = WhatIfService(model, explainer, features)
    return await whatif_service.run_simulation(simulation)

@app.websocket("/api/v1/whatif/ws")
async def whatif_websocket(websocket: WebSocket):
    """Real-time what-if simulation via WebSocket"""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            result = await run_simulation(data)
            await websocket.send_json(result)
    except WebSocketDisconnect:
        pass

# Feedback endpoints
@app.post("/api/v1/feedback/submit")
async def submit_feedback(feedback: FeedbackSubmission, user = Depends(get_current_user)):
    """Submit user feedback"""
    feedback_service = FeedbackService(db)
    return await feedback_service.submit_feedback(feedback)

@app.get("/api/v1/feedback/analytics")
async def feedback_analytics(model_id: int = None, user = Depends(get_current_user)):
    """Get feedback analytics"""
    feedback_service = FeedbackService(db)
    return await feedback_service.get_analytics(model_id)
```

### 3.2 Request/Response Models

```python
from pydantic import BaseModel, Field
from typing import List, Dict

class ExplainRequest(BaseModel):
    model_id: int
    instance: List[float]
    method: str = "shap"  # 'shap' or 'lime'

class WhatIfSimulation(BaseModel):
    model_id: int
    original_instance: List[float]
    modified_features: Dict[str, float]
    session_id: str

class FeedbackSubmission(BaseModel):
    explanation_id: int
    rating: int = Field(..., ge=1, le=5)
    comment: str = ""
    helpful: bool
```

---

## 4. Core Components

### 4.1 Explanation Engine

```python
# backend/explainers/explanation_engine.py
import shap
import lime.lime_tabular

class ExplanationEngine:
    def __init__(self, model, X_train, feature_names):
        self.model = model
        self.X_train = X_train
        self.feature_names = feature_names
        self.shap_explainer = shap.TreeExplainer(model)
        self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            X_train, feature_names=feature_names, mode='classification'
        )
    
    def generate_global_explanation(self):
        """Global feature importance using SHAP"""
        shap_values = self.shap_explainer.shap_values(self.X_train)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        return {
            "type": "global",
            "feature_importance": [
                {"feature": name, "importance": float(imp)}
                for name, imp in zip(self.feature_names, mean_abs_shap)
            ]
        }
    
    def generate_local_explanation(self, instance, method="shap"):
        """Local explanation for single instance"""
        if method == "shap":
            shap_values = self.shap_explainer.shap_values(instance)
            contributions = [
                {"feature": name, "value": float(instance[i]), 
                 "contribution": float(shap_values[i])}
                for i, name in enumerate(self.feature_names)
            ]
        else:  # LIME
            exp = self.lime_explainer.explain_instance(
                instance, self.model.predict_proba, num_features=10
            )
            contributions = [
                {"feature": feat, "contribution": weight}
                for feat, weight in exp.as_list()
            ]
        
        return {
            "type": "local",
            "method": method,
            "prediction": int(self.model.predict([instance])[0]),
            "contributions": sorted(contributions, 
                key=lambda x: abs(x.get("contribution", 0)), reverse=True)
        }
```

### 4.2 Personalization Service

```python
# backend/services/personalization_service.py

ROLE_COMPLEXITY = {
    "developer": 10,  # Full technical details
    "end_user": 2     # Simplified view
}

class PersonalizationService:
    def adapt_explanation(self, explanation, user_role):
        """Adapt explanation based on user role"""
        complexity = ROLE_COMPLEXITY[user_role]
        
        if user_role == "developer":
            return {
                "role": "developer",
                "complexity_level": complexity,
                "explanation": explanation,
                "all_features": explanation.get("contributions", []),
                "show_technical": True
            }
        else:  # end_user
            top_3 = explanation.get("contributions", [])[:3]
            return {
                "role": "end_user",
                "complexity_level": complexity,
                "summary": self._generate_summary(explanation, top_3),
                "key_factors": [
                    {"factor": c["feature"].replace("_", " ").title(),
                     "impact": "high" if abs(c.get("contribution", 0)) > 0.5 else "medium"}
                    for c in top_3
                ],
                "show_technical": False
            }
    
    def _generate_summary(self, explanation, top_contributors):
        """Generate natural language summary"""
        pred = explanation.get("prediction")
        if top_contributors:
            top = top_contributors[0]["feature"].replace("_", " ").title()
            return f"Predicted: {pred}. Main factor: {top}"
        return f"Predicted: {pred}"
```

### 4.3 What-If Service

```python
# backend/services/whatif_service.py

class WhatIfService:
    def __init__(self, model, explainer, feature_names):
        self.model = model
        self.explainer = explainer
        self.feature_names = feature_names
    
    async def run_simulation(self, simulation):
        """Run what-if simulation"""
        # Create modified instance
        modified = np.array(simulation.original_instance)
        for feature, value in simulation.modified_features.items():
            idx = self.feature_names.index(feature)
            modified[idx] = value
        
        # Get predictions
        orig_pred = self.model.predict([simulation.original_instance])[0]
        mod_pred = self.model.predict([modified])[0]
        
        # Get probabilities
        orig_proba = self.model.predict_proba([simulation.original_instance])[0]
        mod_proba = self.model.predict_proba([modified])[0]
        
        return {
            "original": {"prediction": int(orig_pred), "probability": orig_proba.tolist()},
            "modified": {"prediction": int(mod_pred), "probability": mod_proba.tolist()},
            "prediction_changed": orig_pred != mod_pred,
            "confidence_change": float(np.max(mod_proba) - np.max(orig_proba))
        }
```

---

## 5. Database Schema

```sql
-- Users
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(100) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(50) CHECK (role IN ('developer', 'end_user', 'admin')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Models
CREATE TABLE models (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    version VARCHAR(50) NOT NULL,
    framework VARCHAR(50) CHECK (framework IN ('sklearn', 'tensorflow', 'pytorch')),
    file_path TEXT NOT NULL,
    feature_names JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(name, version)
);

-- Explanations
CREATE TABLE explanations (
    id SERIAL PRIMARY KEY,
    model_id INTEGER REFERENCES models(id) ON DELETE CASCADE,
    user_id INTEGER REFERENCES users(id),
    explanation_type VARCHAR(50) CHECK (explanation_type IN ('global', 'local')),
    method VARCHAR(50) CHECK (method IN ('shap', 'lime')),
    input_data JSONB,
    output_data JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Feedback
CREATE TABLE feedback (
    id SERIAL PRIMARY KEY,
    explanation_id INTEGER REFERENCES explanations(id) ON DELETE CASCADE,
    user_id INTEGER REFERENCES users(id),
    rating INTEGER CHECK (rating >= 1 AND rating <= 5),
    comment TEXT,
    helpful BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Sessions (for what-if)
CREATE TABLE sessions (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255) UNIQUE NOT NULL,
    user_id INTEGER REFERENCES users(id),
    model_id INTEGER REFERENCES models(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Simulations
CREATE TABLE simulations (
    id SERIAL PRIMARY KEY,
    session_id VARCHAR(255) REFERENCES sessions(session_id),
    original_instance JSONB NOT NULL,
    modified_features JSONB NOT NULL,
    predictions JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## 6. Deployment

### 6.1 Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: adaptxai
      POSTGRES_USER: adaptxai_user
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  backend:
    build: .
    environment:
      DATABASE_URL: postgresql://adaptxai_user:${DB_PASSWORD}@postgres:5432/adaptxai
      REDIS_URL: redis://redis:6379
    ports:
      - "8000:8000"
    depends_on:
      - postgres
      - redis
    command: uvicorn backend.main:app --host 0.0.0.0 --port 8000

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    environment:
      REACT_APP_API_URL: http://localhost:8000

  streamlit:
    build: ./dashboard
    ports:
      - "8501:8501"
    environment:
      API_URL: http://backend:8000

volumes:
  postgres_data:
```

### 6.2 AWS Deployment

```
Route 53 → CloudFront → ALB → ECS Fargate (Backend + Frontend)
                                    ↓
                        RDS PostgreSQL + ElastiCache Redis + S3 (Models)
```

---

## 7. Security

```python
# backend/security/auth.py
from jose import jwt
from passlib.context import CryptContext

SECRET_KEY = os.getenv("JWT_SECRET")
pwd_context = CryptContext(schemes=["bcrypt"])

def create_access_token(data: dict):
    """Create JWT token"""
    return jwt.encode(data, SECRET_KEY, algorithm="HS256")

async def get_current_user(credentials = Depends(security)):
    """Validate JWT and return user"""
    token = credentials.credentials
    payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
    return {"id": payload["user_id"], "role": payload["role"]}
```

---

## 8. Performance Optimization

```python
# Caching with Redis
from functools import wraps

def cached(prefix: str, ttl: int = 3600):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            cache_key = f"{prefix}:{hash(str(args))}"
            cached_result = redis_client.get(cache_key)
            if cached_result:
                return json.loads(cached_result)
            
            result = await func(*args, **kwargs)
            redis_client.setex(cache_key, ttl, json.dumps(result))
            return result
        return wrapper
    return decorator

@cached(prefix="explanation", ttl=1800)
async def get_explanation(model_id, instance):
    # Expensive operation
    pass
```

---

## 9. Summary

**Architecture:** 6-layer system (Input → Explanation → Personalization → Feedback → What-If → Output)  
**API:** FastAPI with REST + WebSocket  
**Database:** PostgreSQL with 6 core tables  
**Deployment:** Docker Compose / AWS ECS  
**Security:** JWT auth, role-based access  
**Performance:** Redis caching, async processing  

**Status:** Production-ready for AWS AI for Bharat Hackathon
