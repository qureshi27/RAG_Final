from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware



from routes.signup_sign_in import router as signup_sign_in_router

import uvicorn
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

# CORS Middleware added below
origins = ["*"]  # Allows all origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)


app.include_router(signup_sign_in_router)
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

