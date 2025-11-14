from fastapi import FastAPI

app = FastAPI(
    title="Ghost in the Machine API",
    version="1.0",
    description="API for serving the best actual lap and the generated Ghost Lap."
)

@app.get("/")
def read_root():
    return {"status": "API is running"}

# We will add more endpoints here in the next phase
# e.g., /laps/best_actual and /laps/ghost