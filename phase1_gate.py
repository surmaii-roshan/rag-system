from config import Config
from utils.logger import get_logger
from ingest import loader
from retrieve import cache
from generate import groq_client

log = get_logger("phase1-gate")

log.info(f"Primary model: {Config.PRIMARY_MODEL}")
log.info(f"Chunk size: {Config.CHUNK_SIZE}")
log.info("Phase 1 gate: PASSED")