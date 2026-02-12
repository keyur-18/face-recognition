from fastapi import FastAPI,Form,File,UploadFile
import os
import shutil
from utils.embed import register_user,recognition
import tempfile
app  = FastAPI(title="face recognition API")

@app.post("/register")
async def register_face(name:str = Form(...),images : list[UploadFile] = File(...)):
    tmpdir = tempfile.mkdtemp()
    # tmpdir = "temp"

    if len(images) != 3:
        return {"status":"error" ,"message":"exactly 3 images required"}
    try:
        for idx, img in enumerate(images):
            img.file.seek(0)
            file_path = os.path.join(tmpdir, f"{idx+1}.jpg")

            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(img.file, buffer)
        print("Temp dir:", tmpdir)
        print("Files:", os.listdir(tmpdir))
        register_user(name,tmpdir)

        return {
                "status": " success",
                "message": f"{name} registered successfully"
            }
    finally:
        shutil.rmtree(tmpdir)


@app.post("/recognize")
async def recognize_face(
    image: UploadFile = File(...)
):
    tmpdir = tempfile.mkdtemp()
    try:
        path = os.path.join(tmpdir, "final.jpg")
        with open(path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        result =recognition(path)

        return {
            "status": "success",
            "name": result
        }
    finally:
        shutil.rmtree(tmpdir)