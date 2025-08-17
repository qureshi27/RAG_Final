from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import os
from controllers.retrieve import ai_assistant
from controllers.signup_sign_in import (
    signup_user, signin_user, handle_file_embedding_by_email_and_filename,
    chroma_client
)

UPLOAD_FOLDER = "./uploaded_files"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # <-- This line creates the folder!

import shutil

router = APIRouter()

@router.post("/signup")
def signup(email: str = Form(...), password: str = Form(...)):
    try:
        signup_user(email, password)
        return {"message": "User signed up successfully."}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/signin")
def signin(email: str = Form(...), password: str = Form(...)):
    try:
        signin_user(email, password)
        return {"message": "Sign in successful."}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))



@router.post("/upload")
def upload_file(email: str = Form(...), file: UploadFile = File(...)):
    UPLOAD_FOLDER = "./uploaded_files"
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)

    file.file.seek(0)  # <-- This is key!
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        result = handle_file_embedding_by_email_and_filename(email, file.filename, chroma_client)
        return {"message": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




@router.post("/retrieve")
async def retrieve(
    email: str = Form(...), 
    session_id: str = Form(...), 
    query: str = Form(...)
):
    try:
        result = await ai_assistant(query, email, session_id)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
