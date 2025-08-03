import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api import router as api_router
from api.context import AppContext

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("FRONTEND_URL"), "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Run-Id"],
)

# Shared context (loaded once)
app.state.ctx = AppContext()

# Mount routes
app.include_router(api_router)
