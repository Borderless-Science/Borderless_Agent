from fastapi import FastAPI
from pydantic import BaseModel
from aggg import crew  # Import your existing agent

app = FastAPI(title="Agent API")

class QueryRequest(BaseModel):
    query: str

@app.get("/")
def home():
    return {"message": "Agent API is running"}

@app.post("/api/query")
def query_agent(request: QueryRequest):
    """Simple endpoint that calls your existing agent"""
    try:
        # This calls YOUR existing agent code directly
        result = crew.kickoff(inputs={"query": request.query})
        return {"success": True, "result": result.raw}
    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__fast__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)