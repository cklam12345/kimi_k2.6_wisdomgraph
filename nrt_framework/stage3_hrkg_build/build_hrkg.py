"""
Stage 3 — Hyper-Relational Knowledge Graph Construction (LLHKG)
Paper §3.2: Employ LLHKG to generate hyper-relational structures supporting
complex many-to-many relations for long-horizon coding and swarm orchestration.
Target: 1.2 Billion nodes, 14.5 GB (Table 1).

PROOF TASK:
  Lift flat SPO triples into hyper-relational statements:
    (subject, predicate, object) + {qualifier_key: qualifier_value, ...}
  Store in Neo4j with the DIKW node labels:
    :Data  :Information  :Knowledge  :Wisdom
"""
import pickle, json
from pathlib import Path
from typing import Generator
from neo4j import GraphDatabase
from tqdm import tqdm
from dataclasses import dataclass


@dataclass
class HyperTriple:
    subject: str
    predicate: str
    object: str
    qualifiers: dict          # e.g. {"domain": "code", "confidence": 0.91, "layer": 42}
    dikw_level: str           # "Data" | "Information" | "Knowledge" | "Wisdom"
    temporal_weight: float    # 1.0 = fresh; decays toward 0 via §5 temporal decay


DIKW_RULES = {
    "Data":        lambda t: t.qualifiers.get("confidence", 0) < 0.6,
    "Information": lambda t: 0.6  <= t.qualifiers.get("confidence", 0) < 0.75,
    "Knowledge":   lambda t: 0.75 <= t.qualifiers.get("confidence", 0) < 0.90,
    "Wisdom":      lambda t: t.qualifiers.get("confidence", 0) >= 0.90,
}


def classify_dikw(t: HyperTriple) -> str:
    for level, rule in DIKW_RULES.items():
        if rule(t):
            return level
    return "Data"


def spo_to_hyper(spo_triples: list) -> list[HyperTriple]:
    """
    Lift SPO triples into hyper-relational form by attaching qualifiers.
    Each qualifier captures: domain, confidence, source_layer, source_expert,
    timestamp (for temporal decay), and relation_type.
    """
    hyper = []
    for t in spo_triples:
        qualifiers = {
            "domain":        t.domain,
            "confidence":    t.confidence,
            "source_layer":  t.source_layer,
            "source_expert": t.source_expert,
        }
        ht = HyperTriple(
            subject=t.subject,
            predicate=t.predicate,
            object=t.object,
            qualifiers=qualifiers,
            dikw_level="",
            temporal_weight=1.0,
        )
        ht.dikw_level = classify_dikw(ht)
        hyper.append(ht)
    return hyper


CYPHER_MERGE_TRIPLE = """
MERGE (s:Entity {name: $subject})
  SET s.dikw = $dikw_level
MERGE (o:Entity {name: $object})
MERGE (s)-[r:{predicate} {{
  confidence:    $confidence,
  domain:        $domain,
  source_layer:  $source_layer,
  source_expert: $source_expert,
  temporal_weight: $temporal_weight,
  dikw_level:    $dikw_level
}}]->(o)
"""

def ingest_to_neo4j(
    driver,
    hyper_triples: list[HyperTriple],
    batch_size: int = 1000,
):
    """Batch-ingest hyper-triples into Neo4j using parameterised Cypher."""
    with driver.session() as session:
        for i in range(0, len(hyper_triples), batch_size):
            batch = hyper_triples[i:i+batch_size]
            tx_data = [
                {
                    "subject":        ht.subject,
                    "predicate":      ht.predicate,
                    "object":         ht.object,
                    "dikw_level":     ht.dikw_level,
                    "confidence":     ht.qualifiers["confidence"],
                    "domain":         ht.qualifiers["domain"],
                    "source_layer":   ht.qualifiers["source_layer"],
                    "source_expert":  ht.qualifiers["source_expert"],
                    "temporal_weight":ht.temporal_weight,
                }
                for ht in batch
            ]
            session.run(
                "UNWIND $rows AS row " +
                "MERGE (s:Entity {name: row.subject}) "
                "MERGE (o:Entity {name: row.object}) "
                "MERGE (s)-[r:RELATES_TO {predicate: row.predicate}]->(o) "
                "SET r += {confidence: row.confidence, domain: row.domain, "
                "source_layer: row.source_layer, temporal_weight: row.temporal_weight, "
                "dikw_level: row.dikw_level}",
                rows=tx_data,
            )


def run(config_path: str = "../config/pipeline.yaml"):
    import yaml
    cfg = yaml.safe_load(Path(config_path).read_text())
    spo_dir = Path("..") / cfg["stage2_spo"]["output_dir"]
    neo4j_uri  = cfg["stage3_hrkg"]["neo4j_uri"]
    neo4j_user = cfg["stage3_hrkg"]["neo4j_user"]
    neo4j_pass = cfg["stage3_hrkg"]["neo4j_password"]

    driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_pass))

    shard_files = sorted(spo_dir.glob("shard_*.pkl"))
    total_nodes = 0
    for shard_path in tqdm(shard_files, desc="Building HRKG"):
        with open(shard_path, "rb") as f:
            spo_batch = pickle.load(f)
        hyper = spo_to_hyper(spo_batch)
        ingest_to_neo4j(driver, hyper)
        total_nodes += len(hyper)

    driver.close()
    print(f"Stage 3 complete. ~{total_nodes} hyper-triples ingested into Neo4j at {neo4j_uri}")


if __name__ == "__main__":
    import sys
    run(sys.argv[1] if len(sys.argv) > 1 else "../config/pipeline.yaml")
