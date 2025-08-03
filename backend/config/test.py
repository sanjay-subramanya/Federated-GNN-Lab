async def delete_run(path):
    run_id = req.run_id.strip()
    logger.info(f"DEBUG: Received delete request for runId: {run_id}")
    try:
        # delete_run_from_blob(run_id)

        # Also delete local copy if exists
        run_folder = Config.model_dir / run_id
        if run_folder.exists():
            shutil.rmtree(run_folder)
        logger.info(f"DEBUG: Deleted local folder for runId: {run_id}")
        return {"status": "deleted"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# python -m backend.main
# uvicorn router:app --reload
# npm run dev
# npm install react-plotly.js plotly.js
# npm install --save-dev @types/react-plotly.js