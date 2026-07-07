# week-06/app/dependencies/model_loader.py

from fastapi import Request, HTTPException

def get_pipeline(request: Request):
    """
    Ambil pipeline dari app.state.
    Raise 503 jika belum tersedia — bukan 500.
    503 = server sementara tidak bisa melayani (model belum siap)
    500 = bug di kode kita
    """
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Pipeline not available. Server may still be starting up."
        )
    return pipeline