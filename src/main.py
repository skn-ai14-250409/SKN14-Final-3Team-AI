from fastapi import FastAPI

from src.api.router import router

app = FastAPI()

app.include_router(router)

@app.get("/")
def root():
    return {"message": "Hello — root is working"}