from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, Query
from queues.workers import process_query
from client.rq_client import celery_app


app = FastAPI()

@app.get("/")
def root():
    return {"status": "Server is up and running"}

@app.post("/chat")
def chat(
        query: str = Query(..., description="The chat query of user")
):
    job = process_query.delay(query)

    return {"status": "queued", "job_id": job.id}

@app.get("/job-status")
def get_result(
    job_id: str = Query(..., description="Job ID")
):
    job = celery_app.AsyncResult(job_id)
    return {"status": job.status, "result": job.result}