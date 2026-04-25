"""
Stage 3 — Hyper-Relational Knowledge Graph Construction in Neo4j
Paper §3.2: Lift SPO triples into DIKW-labelled hyper-relational graph.

Reads: artifacts/spo_triples/*.jsonl  (output of kimi_extractor.py)
Writes: Neo4j graph (bolt://localhost:7687)

Run:
    python3 build_hrkg.py [config_path]
"""
import json, re
from pathlib import Path
from datetime import datetime
from neo4j import GraphDatabase
from tqdm import tqdm


# Safe relationship type: Neo4j rel types must be alphanumeric + underscore
def safe_rel_type(pred: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", pred.strip().upper().replace(" ", "_"))[:50]


CYPHER_INGEST = """
UNWIND $rows AS row
MERGE (s:Entity {name: row.subj})
  ON CREATE SET s.created = row.extracted_at
  SET s.dikw = row.dikw
MERGE (o:Entity {name: row.obj})
  ON CREATE SET o.created = row.extracted_at
MERGE (s)-[r:RELATES_TO {predicate: row.pred}]->(o)
  SET r.confidence     = row.conf,
      r.dikw           = row.dikw,
      r.source_topic   = row.source_topic,
      r.model          = row.model,
      r.extracted_at   = row.extracted_at,
      r.temporal_weight = 1.0
"""

CYPHER_INDEX = """
CREATE INDEX entity_name IF NOT EXISTS FOR (n:Entity) ON (n.name)
"""


def load_jsonl(spo_dir: Path) -> list[dict]:
    triples = []
    for f in sorted(spo_dir.glob("spo_*.jsonl")):
        if f.stat().st_size == 0:
            continue
        for line in f.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    t = json.loads(line)
                    # Normalise predicate to safe rel type
                    t["pred_safe"] = safe_rel_type(t.get("pred", "RELATES_TO"))
                    triples.append(t)
                except json.JSONDecodeError:
                    pass
    return triples


def ingest(driver, triples: list[dict], batch_size: int = 500):
    with driver.session() as session:
        session.run(CYPHER_INDEX)
        for i in tqdm(range(0, len(triples), batch_size), desc="Ingesting"):
            batch = triples[i:i + batch_size]
            session.run(CYPHER_INGEST, rows=batch)


def stats(driver):
    with driver.session() as session:
        nodes = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        rels  = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
    return nodes, rels


def run(config_path: str = "../config/pipeline.yaml"):
    import yaml
    cfg      = yaml.safe_load(Path(config_path).read_text())
    spo_dir  = Path("..") / cfg["stage2_spo"]["output_dir"]
    uri      = cfg["stage3_hrkg"]["neo4j_uri"]
    user     = cfg["stage3_hrkg"]["neo4j_user"]
    password = cfg["stage3_hrkg"]["neo4j_password"]

    print(f"[Stage3] Loading triples from {spo_dir}")
    triples = load_jsonl(spo_dir)
    print(f"[Stage3] {len(triples)} triples loaded")

    driver = GraphDatabase.driver(uri, auth=(user, password))
    ingest(driver, triples)

    n, r = stats(driver)
    driver.close()

    print(f"\n[Stage3] Done.")
    print(f"  Nodes: {n:,}")
    print(f"  Relationships: {r:,}")
    print(f"  Neo4j: {uri}")


if __name__ == "__main__":
    import sys
    run(sys.argv[1] if len(sys.argv) > 1 else "../config/pipeline.yaml")
