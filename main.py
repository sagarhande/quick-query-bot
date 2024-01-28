from fastapi import FastAPI
import uvicorn

import views

app = FastAPI()

# Include routes from the app module
app.include_router(views.router, prefix="/qqb", tags=["app"])

if __name__ == "__main__":
    uvicorn.run(app)
